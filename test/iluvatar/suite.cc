// SPDX-License-Identifier: Apache-2.0
// Iluvatar FULL-mode correctness suite.
// Covers registration, synchronous read/write, synchronous batch,
// asynchronous batch (submit/wait/cancel/destroy), stream-ordered I/O,
// and the host-buffer (device_id < 0) path.
//
// Requires phoenixfs loaded in FULL mode (phxfs_map_mode=0) and a file
// on a filesystem that supports O_DIRECT. Device ordinal defaults to 0.
//
//   ./test_iluvatar_suite <file> [device]
//   PHX_TEST_FILE=<file> ./test_iluvatar_suite [device]

#include <cuda_runtime.h>

#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include "phoenix.h"

namespace {

int g_run;
int g_fail;

void check(bool ok, const char *fmt, ...)
{
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	g_run++;
	if (ok) {
		printf("  [PASS] %s\n", buf);
	} else {
		g_fail++;
		printf("  [FAIL] %s\n", buf);
	}
}

void *page_alloc(size_t n)
{
	void *p = nullptr;
	if (posix_memalign(&p, 4096, n) != 0)
		return nullptr;
	return p;
}

void fill_bytes(unsigned char *p, size_t n, unsigned seed)
{
	for (size_t i = 0; i < n; i++)
		p[i] = static_cast<unsigned char>(seed + i);
}

bool same_bytes(const unsigned char *p, size_t n, unsigned seed)
{
	for (size_t i = 0; i < n; i++) {
		unsigned char expect = static_cast<unsigned char>(seed + i);
		if (p[i] != expect) {
			printf("    mismatch at %zu got %02x expect %02x\n",
			       i, p[i], expect);
			return false;
		}
	}
	return true;
}

int open_direct(const char *path, off_t bytes)
{
	int fd = open(path, O_CREAT | O_RDWR, 0644);
	if (fd < 0)
		return -1;
	if (posix_fallocate(fd, 0, bytes) != 0) {
		close(fd);
		return -1;
	}
	close(fd);
	return open(path, O_RDWR | O_DIRECT);
}

bool cuda_ok(cudaError_t rc, const char *what)
{
	if (rc == cudaSuccess)
		return true;
	printf("    %s: %s\n", what, cudaGetErrorString(rc));
	return false;
}

/* Iluvatar cudaMemcpy can return before a CPU load of the BAR mapping
 * sees the bytes. phxfs_read/write are pread/pwrite on that mapping. */
bool gpu_visible()
{
	return cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

struct GpuMem {
	void *ptr;
	size_t bytes;
	int dev;
	bool registered;

	GpuMem() : ptr(nullptr), bytes(0), dev(0), registered(false) {}

	bool alloc(int device, size_t n)
	{
		dev = device;
		bytes = n;
		if (!cuda_ok(cudaMalloc(&ptr, n), "cudaMalloc"))
			return false;
		return cuda_ok(cudaMemset(ptr, 0, n), "cudaMemset") &&
		       cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
	}

	bool reg()
	{
		void *target = nullptr;
		int rc = phxfs_regmem(dev, ptr, bytes, &target);
		registered = (rc == 0);
		return registered;
	}

	void release()
	{
		if (registered) {
			phxfs_deregmem(dev, ptr, bytes);
			registered = false;
		}
		if (ptr) {
			cudaFree(ptr);
			ptr = nullptr;
		}
	}

	~GpuMem() { release(); }
};

int host_direct(int fd)
{
	printf("\n=== host buffer (device_id < 0), O_DIRECT ===\n");
	const size_t n = 64 * 1024;
	const off_t off = 0;
	unsigned char *src = static_cast<unsigned char *>(page_alloc(n));
	unsigned char *dst = static_cast<unsigned char *>(page_alloc(n));
	check(src && dst, "aligned host buffers");
	if (!src || !dst) {
		free(src);
		free(dst);
		return -1;
	}
	fill_bytes(src, n, 0x11);
	memset(dst, 0x5a, n);

	ssize_t wr = phxfs_write(fd, -1, src, 0, n, off);
	check(wr == (ssize_t)n, "host sync write %zd", wr);
	ssize_t rd = phxfs_read(fd, -1, dst, 0, n, off);
	check(rd == (ssize_t)n && same_bytes(dst, n, 0x11), "host sync read");

	const int k = 4;
	const size_t chunk = 16 * 1024;
	unsigned char *bsrc = static_cast<unsigned char *>(page_alloc(k * chunk));
	unsigned char *bdst = static_cast<unsigned char *>(page_alloc(k * chunk));
	phxfs_io_req_t wreq[4];
	phxfs_io_req_t rreq[4];
	memset(wreq, 0, sizeof wreq);
	memset(rreq, 0, sizeof rreq);
	for (int i = 0; i < k; i++) {
		fill_bytes(bsrc + i * chunk, chunk, 0x40 + i);
		wreq[i].fd = fd;
		wreq[i].device_id = -1;
		wreq[i].buf = bsrc;
		wreq[i].buf_offset = i * chunk;
		wreq[i].nbytes = chunk;
		wreq[i].f_offset = 256 * 1024 + i * chunk;
		wreq[i].result = -1;
	}
	int rc = phxfs_write_batch(wreq, k);
	check(rc == 0, "host sync batch write rc=%d", rc);
	bool wdata = true;
	for (int i = 0; i < k; i++)
		wdata = wdata && wreq[i].result == (ssize_t)chunk;
	check(wdata, "host sync batch write results");

	for (int i = 0; i < k; i++) {
		rreq[i].fd = fd;
		rreq[i].device_id = -1;
		rreq[i].buf = bdst;
		rreq[i].buf_offset = i * chunk;
		rreq[i].nbytes = chunk;
		rreq[i].f_offset = wreq[i].f_offset;
		rreq[i].result = -1;
	}
	rc = phxfs_read_batch(rreq, k);
	bool rdata = rc == 0;
	for (int i = 0; i < k; i++)
		rdata = rdata && rreq[i].result == (ssize_t)chunk &&
			same_bytes(bdst + i * chunk, chunk, 0x40 + i);
	check(rdata, "host sync batch read rc=%d", rc);

	phxfs_batch_t *async = phxfs_batch_submit_read(rreq, k);
	check(async != nullptr, "host async submit");
	if (async) {
		int wrv = phxfs_batch_wait(async);
		bool ok = wrv == 0;
		for (int i = 0; i < k; i++)
			ok = ok && rreq[i].result == (ssize_t)chunk;
		check(ok, "host async wait rc=%d", wrv);
	}

	check(phxfs_batch_cancel(nullptr) == -EINVAL, "cancel NULL");
	check(phxfs_batch_wait(nullptr) == -EINVAL, "wait NULL");
	check(phxfs_batch_destroy(nullptr) == -EINVAL, "destroy NULL");
	phxfs_batch_t *empty = phxfs_batch_submit_read(nullptr, 0);
	check(empty && phxfs_batch_wait(empty) == 0, "empty async batch");

	free(src);
	free(dst);
	free(bsrc);
	free(bdst);
	return 0;
}

bool gpu_roundtrip(int fd, int dev, size_t n, off_t foff, unsigned seed, const char *tag)
{
	GpuMem gpu;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(n));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(n));
	if (!host || !back || !gpu.alloc(dev, n) || !gpu.reg()) {
		check(false, "%s setup", tag);
		free(host);
		free(back);
		return false;
	}
	fill_bytes(host, n, seed);
	bool h2d = cuda_ok(cudaMemcpy(gpu.ptr, host, n, cudaMemcpyHostToDevice), "H2D") &&
		   gpu_visible();
	ssize_t wr = phxfs_write(fd, dev, gpu.ptr, 0, n, foff);
	memset(back, 0, n);
	bool file_ok = pread(fd, back, n, foff) == (ssize_t)n && same_bytes(back, n, seed);
	memset(back, 0, n);
	bool cleared = cuda_ok(cudaMemset(gpu.ptr, 0, n), "clear") && gpu_visible();
	ssize_t rd = phxfs_read(fd, dev, gpu.ptr, 0, n, foff);
	bool back_ok = gpu_visible() &&
		       cuda_ok(cudaMemcpy(back, gpu.ptr, n, cudaMemcpyDeviceToHost), "D2H");
	check(h2d && file_ok && cleared && wr == (ssize_t)n && rd == (ssize_t)n &&
	      back_ok && same_bytes(back, n, seed),
	      "%s sync %zu bytes write=%zd read=%zd", tag, n, wr, rd);
	free(host);
	free(back);
	return true;
}

void gpu_regmem(int dev)
{
	printf("\n=== registration ===\n");
	uint64_t page = phxfs_get_page_size();
	check(page == 4096, "page size %llu", (unsigned long long)page);
	check(phxfs_get_map_mode(dev) == 0, "FULL map mode");

	GpuMem a;
	void *target = nullptr;
	check(a.alloc(dev, 64 * 1024), "alloc 64KiB");
	int rc = phxfs_regmem(dev, a.ptr, a.bytes, &target);
	a.registered = (rc == 0);
	check(rc == 0 && target != nullptr, "regmem 64KiB rc=%d", rc);
	void *target2 = nullptr;
	rc = phxfs_regmem(dev, a.ptr, a.bytes, &target2);
	check(rc == 0, "exact duplicate regmem rc=%d", rc);
	check(phxfs_deregmem(dev, a.ptr, a.bytes) == 0, "dereg one ref");
	check(phxfs_deregmem(dev, a.ptr, a.bytes) == 0, "dereg last ref");
	a.registered = false;
	check(phxfs_deregmem(dev, a.ptr, a.bytes) != 0, "double deregister fails");

	rc = phxfs_regmem(dev, a.ptr, a.bytes, &target);
	a.registered = (rc == 0);
	check(rc == 0, "reuse after dereg rc=%d", rc);

	GpuMem b;
	check(b.alloc(dev, 32 * 1024), "alloc overlap probe");
	/* Same device VA cannot be split from userspace; overlap is a second
	 * registration whose range intersects an existing one. A shorter
	 * prefix of the same allocation is that case. */
	rc = phxfs_regmem(dev, a.ptr, 32 * 1024, &target2);
	check(rc == -EINVAL, "overlapping regmem rc=%d", rc);

	rc = phxfs_regmem(dev, a.ptr, page + 1, &target2);
	check(rc == -EINVAL, "unaligned length rc=%d", rc);

	GpuMem raw;
	check(raw.alloc(dev, 4096), "alloc unregistered");
	unsigned char *tmp = static_cast<unsigned char *>(page_alloc(4096));
	ssize_t io = -1;
	if (tmp && raw.ptr) {
		memset(tmp, 0, 4096);
		io = phxfs_read(-1, dev, raw.ptr, 0, 4096, 0);
	}
	check(io < 0, "unregistered GPU read rc=%zd", io);
	free(tmp);
}

void gpu_partial(int fd, int dev)
{
	printf("\n=== partial offset ===\n");
	const size_t total = 16 * 1024;
	const size_t n = 4096;
	const off_t buf_off = 4096;
	const off_t foff = 8 * 1024 * 1024;
	GpuMem gpu;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(total));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(n));
	if (!host || !back || !gpu.alloc(dev, total) || !gpu.reg()) {
		check(false, "partial setup");
		free(host);
		free(back);
		return;
	}
	memset(host, 0, total);
	fill_bytes(host + buf_off, n, 0x90);
	bool h2d = cuda_ok(cudaMemcpy(gpu.ptr, host, total, cudaMemcpyHostToDevice), "H2D") &&
		   gpu_visible();
	ssize_t wr = phxfs_write(fd, dev, gpu.ptr, buf_off, n, foff);
	bool clr = cuda_ok(cudaMemset(gpu.ptr, 0, total), "clear") && gpu_visible();
	ssize_t rd = phxfs_read(fd, dev, gpu.ptr, buf_off, n, foff);
	bool d2h = gpu_visible() &&
		   cuda_ok(cudaMemcpy(back, static_cast<char *>(gpu.ptr) + buf_off,
				      n, cudaMemcpyDeviceToHost), "D2H");
	check(h2d && clr && d2h && wr == (ssize_t)n && rd == (ssize_t)n &&
	      same_bytes(back, n, 0x90),
	      "4KiB at buf_offset 4KiB write=%zd read=%zd", wr, rd);
	free(host);
	free(back);
}

void gpu_batch(int fd, int dev, bool async_api, off_t fbase, const char *tag)
{
	const int k = 4;
	const size_t chunk = 64 * 1024;
	const size_t total = k * chunk;
	GpuMem src;
	GpuMem dst;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(total));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(total));
	if (!host || !back || !src.alloc(dev, total) || !dst.alloc(dev, total) ||
	    !src.reg() || !dst.reg()) {
		check(false, "%s setup", tag);
		free(host);
		free(back);
		return;
	}
	for (int i = 0; i < k; i++)
		fill_bytes(host + i * chunk, chunk, 0x20 + i);
	if (!cuda_ok(cudaMemcpy(src.ptr, host, total, cudaMemcpyHostToDevice), "H2D") ||
	    !gpu_visible()) {
		check(false, "%s H2D", tag);
		free(host);
		free(back);
		return;
	}

	phxfs_io_req_t wreq[4];
	phxfs_io_req_t rreq[4];
	memset(wreq, 0, sizeof wreq);
	memset(rreq, 0, sizeof rreq);
	for (int i = 0; i < k; i++) {
		wreq[i].fd = fd;
		wreq[i].device_id = dev;
		wreq[i].buf = src.ptr;
		wreq[i].buf_offset = i * chunk;
		wreq[i].nbytes = chunk;
		wreq[i].f_offset = fbase + i * chunk;
		wreq[i].result = -1;
		rreq[i] = wreq[i];
		rreq[i].buf = dst.ptr;
		rreq[i].result = -1;
	}

	bool ok = true;
	if (!async_api) {
		int rc = phxfs_write_batch(wreq, k);
		ok = rc == 0;
		for (int i = 0; i < k; i++)
			ok = ok && wreq[i].result == (ssize_t)chunk;
		rc = phxfs_read_batch(rreq, k);
		ok = ok && rc == 0;
		for (int i = 0; i < k; i++)
			ok = ok && rreq[i].result == (ssize_t)chunk;
	} else {
		phxfs_batch_t *w = phxfs_batch_submit_write(wreq, k);
		phxfs_batch_t *r = nullptr;
		int wrc = w ? phxfs_batch_wait(w) : -1;
		ok = w && wrc == 0;
		for (int i = 0; i < k; i++)
			ok = ok && wreq[i].result == (ssize_t)chunk;
		r = phxfs_batch_submit_read(rreq, k);
		int rrc = r ? phxfs_batch_wait(r) : -1;
		ok = ok && r && rrc == 0;
		for (int i = 0; i < k; i++)
			ok = ok && rreq[i].result == (ssize_t)chunk;
	}
	if (!gpu_visible() ||
	    !cuda_ok(cudaMemcpy(back, dst.ptr, total, cudaMemcpyDeviceToHost), "D2H"))
		ok = false;
	for (int i = 0; i < k; i++)
		ok = ok && same_bytes(back + i * chunk, chunk, 0x20 + i);
	check(ok, "%s 4x64KiB", tag);
	free(host);
	free(back);
}

void gpu_async_pipeline(int fd, int dev)
{
	printf("\n=== async batches in flight ===\n");
	const size_t n = 64 * 1024;
	const off_t base = 14 * 1024 * 1024;
	GpuMem a, b;
	unsigned char *ha = static_cast<unsigned char *>(page_alloc(n));
	unsigned char *hb = static_cast<unsigned char *>(page_alloc(n));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(n));
	if (!ha || !hb || !back || !a.alloc(dev, n) || !b.alloc(dev, n) ||
	    !a.reg() || !b.reg()) {
		check(false, "pipeline setup");
		free(ha);
		free(hb);
		free(back);
		return;
	}
	fill_bytes(ha, n, 0xa1);
	fill_bytes(hb, n, 0xb2);
	cudaMemcpy(a.ptr, ha, n, cudaMemcpyHostToDevice);
	cudaMemcpy(b.ptr, hb, n, cudaMemcpyHostToDevice);
	gpu_visible();

	phxfs_io_req_t ra, rb;
	memset(&ra, 0, sizeof ra);
	memset(&rb, 0, sizeof rb);
	ra.fd = rb.fd = fd;
	ra.device_id = rb.device_id = dev;
	ra.buf = a.ptr;
	rb.buf = b.ptr;
	ra.nbytes = rb.nbytes = n;
	ra.f_offset = base;
	rb.f_offset = base + n;
	ra.result = rb.result = -1;

	phxfs_batch_t *wa = phxfs_batch_submit_write(&ra, 1);
	phxfs_batch_t *wb = phxfs_batch_submit_write(&rb, 1);
	int rca = wa ? phxfs_batch_wait(wa) : -1;
	int rcb = wb ? phxfs_batch_wait(wb) : -1;
	check(rca == 0 && rcb == 0 && ra.result == (ssize_t)n && rb.result == (ssize_t)n,
	      "two in-flight writes %d/%d", rca, rcb);

	cudaMemset(a.ptr, 0, n);
	cudaMemset(b.ptr, 0, n);
	gpu_visible();
	ra.result = rb.result = -1;
	phxfs_batch_t *qa = phxfs_batch_submit_read(&ra, 1);
	phxfs_batch_t *qb = phxfs_batch_submit_read(&rb, 1);
	rca = qa ? phxfs_batch_wait(qa) : -1;
	rcb = qb ? phxfs_batch_wait(qb) : -1;
	gpu_visible();
	cudaMemcpy(back, a.ptr, n, cudaMemcpyDeviceToHost);
	bool aok = same_bytes(back, n, 0xa1);
	cudaMemcpy(back, b.ptr, n, cudaMemcpyDeviceToHost);
	bool bok = same_bytes(back, n, 0xb2);
	check(rca == 0 && rcb == 0 && aok && bok, "two in-flight reads");
	free(ha);
	free(hb);
	free(back);
}

void gpu_cancel_and_destroy(int fd, int dev)
{
	printf("\n=== cancel and destroy ===\n");
	const size_t n = 64 * 1024;
	GpuMem gpu;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(n));
	if (!host || !gpu.alloc(dev, n) || !gpu.reg()) {
		check(false, "cancel setup");
		free(host);
		return;
	}
	fill_bytes(host, n, 0x33);
	cudaMemcpy(gpu.ptr, host, n, cudaMemcpyHostToDevice);
	gpu_visible();

	phxfs_io_req_t req;
	memset(&req, 0, sizeof req);
	req.fd = fd;
	req.device_id = dev;
	req.buf = gpu.ptr;
	req.nbytes = n;
	req.f_offset = 16 * 1024 * 1024;
	req.result = -1;
	phxfs_batch_t *h = phxfs_batch_submit_write(&req, 1);
	int crc = h ? phxfs_batch_cancel(h) : -1;
	int wrc = h ? phxfs_batch_wait(h) : -1;
	bool result_ok = req.result == (ssize_t)n || req.result == -ECANCELED;
	check(crc == 0 && wrc >= 0 && result_ok,
	      "cancel then wait crc=%d wrc=%d result=%zd", crc, wrc, req.result);

	req.result = -1;
	req.f_offset = 17 * 1024 * 1024;
	h = phxfs_batch_submit_write(&req, 1);
	int drc = h ? phxfs_batch_destroy(h) : -1;
	check(drc == 0, "destroy in-flight batch rc=%d", drc);

	req.f_offset = 18 * 1024 * 1024;
	req.result = -1;
	ssize_t wr = phxfs_write(fd, dev, gpu.ptr, 0, n, req.f_offset);
	cudaMemset(gpu.ptr, 0, n);
	gpu_visible();
	ssize_t rd = phxfs_read(fd, dev, gpu.ptr, 0, n, req.f_offset);
	gpu_visible();
	unsigned char *back = static_cast<unsigned char *>(page_alloc(n));
	bool d2h = back && cuda_ok(cudaMemcpy(back, gpu.ptr, n, cudaMemcpyDeviceToHost), "D2H");
	check(wr == (ssize_t)n && rd == (ssize_t)n && d2h && back &&
	      same_bytes(back, n, 0x33),
	      "sync I/O after destroy write=%zd read=%zd", wr, rd);
	free(host);
	free(back);
}

void gpu_stream(int fd, int dev)
{
	printf("\n=== stream ===\n");
	const int k = 4;
	const size_t chunk = 64 * 1024;
	const size_t total = k * chunk;
	const off_t fbase = 20 * 1024 * 1024;
	GpuMem gpu;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(total));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(total));
	cudaStream_t stream = nullptr;
	if (!host || !back || !gpu.alloc(dev, total) || !gpu.reg() ||
	    !cuda_ok(cudaStreamCreateWithFlags(&stream, 0), "stream create")) {
		check(false, "stream setup");
		free(host);
		free(back);
		return;
	}
	for (int i = 0; i < k; i++)
		fill_bytes(host + i * chunk, chunk, 0x70 + i);
	cudaMemcpy(gpu.ptr, host, total, cudaMemcpyHostToDevice);

	size_t nbytes[4];
	off_t boff[4];
	off_t foff[4];
	ssize_t done[4];
	bool submitted = true;
	for (int i = 0; i < k; i++) {
		nbytes[i] = chunk;
		boff[i] = i * chunk;
		foff[i] = fbase + i * chunk;
		done[i] = -1;
		int rc = phxfs_write_stream(fd, gpu.ptr, &nbytes[i], &boff[i],
					     &foff[i], &done[i], stream);
		if (rc != 0)
			submitted = false;
	}
	bool synced = cuda_ok(cudaStreamSynchronize(stream), "sync writes");
	bool wrote = submitted && synced;
	for (int i = 0; i < k; i++)
		wrote = wrote && done[i] == (ssize_t)chunk;
	check(wrote, "stream writes");

	cudaMemset(gpu.ptr, 0, total);
	submitted = true;
	for (int i = 0; i < k; i++) {
		done[i] = -1;
		int rc = phxfs_read_stream(fd, gpu.ptr, &nbytes[i], &boff[i],
					    &foff[i], &done[i], stream);
		if (rc != 0)
			submitted = false;
	}
	synced = cuda_ok(cudaStreamSynchronize(stream), "sync reads");
	bool d2h = cuda_ok(cudaMemcpy(back, gpu.ptr, total, cudaMemcpyDeviceToHost), "D2H");
	bool read_ok = submitted && synced && d2h;
	for (int i = 0; i < k; i++)
		read_ok = read_ok && done[i] == (ssize_t)chunk &&
			  same_bytes(back + i * chunk, chunk, 0x70 + i);
	check(read_ok, "stream reads");
	cudaStreamSynchronize(stream);
	free(host);
	free(back);
	(void)stream;
}

/* 256MiB crosses more than one KMD sg_lm chunk, so a wrong
 * itr_address_range stride shows up here and nowhere else. Runs last
 * because it overwrites the whole file. */
void gpu_large_chunk(int fd, int dev)
{
	printf("\n=== 256MiB multi-chunk ===\n");
	const size_t n = 256UL * 1024 * 1024;
	GpuMem gpu;
	unsigned char *host = static_cast<unsigned char *>(page_alloc(n));
	unsigned char *back = static_cast<unsigned char *>(page_alloc(n));
	if (!host || !back || !gpu.alloc(dev, n) || !gpu.reg()) {
		check(false, "256MiB setup");
		free(host);
		free(back);
		return;
	}
	fill_bytes(host, n, 0x5a);
	bool h2d = cuda_ok(cudaMemcpy(gpu.ptr, host, n, cudaMemcpyHostToDevice), "H2D") &&
		   gpu_visible();
	ssize_t wr = phxfs_write(fd, dev, gpu.ptr, 0, n, 0);
	memset(back, 0, n);
	bool file_ok = pread(fd, back, n, 0) == (ssize_t)n && same_bytes(back, n, 0x5a);
	memset(back, 0, n);
	bool cleared = cuda_ok(cudaMemset(gpu.ptr, 0, n), "clear") && gpu_visible();
	ssize_t rd = phxfs_read(fd, dev, gpu.ptr, 0, n, 0);
	bool back_ok = gpu_visible() &&
		       cuda_ok(cudaMemcpy(back, gpu.ptr, n, cudaMemcpyDeviceToHost), "D2H");
	check(h2d && file_ok && cleared && wr == (ssize_t)n && rd == (ssize_t)n &&
	      back_ok && same_bytes(back, n, 0x5a),
	      "256MiB multi-chunk write=%zd read=%zd", wr, rd);
	free(host);
	free(back);
}

extern "C" const char *iluvatar_libcuda_path(void);

bool libcuda_matches(void)
{
	const char *want = iluvatar_libcuda_path();
	FILE *maps = fopen("/proc/self/maps", "r");
	char line[512];
	if (!want || !maps)
		return false;
	while (fgets(line, sizeof line, maps)) {
		if (!strstr(line, "libcuda.so"))
			continue;
		bool ok = strstr(line, want) != nullptr;
		fclose(maps);
		if (!ok)
			printf("    loaded libcuda is not %s\n%s", want, line);
		return ok;
	}
	fclose(maps);
	return false;
}

}  // namespace

int main(int argc, char **argv)
{
	const char *path = argc > 1 ? argv[1] : getenv("PHX_TEST_FILE");
	int dev = argc > 2 ? atoi(argv[2]) : 0;
	if (!path || !path[0]) {
		fprintf(stderr, "usage: %s <odirect-file> [device]\n", argv[0]);
		return 2;
	}
	/* 256MiB for the multi-chunk case; every other case uses offsets below
	 * 32MiB and does not care about the extra length. */
	const off_t file_bytes = 256 * 1024 * 1024;

	printf("file %s device %d\n", path, dev);
	int fd = open_direct(path, file_bytes);
	check(fd >= 0, "O_DIRECT file (%s)", path);
	if (fd < 0)
		return 1;

	host_direct(fd);

	printf("\n=== device ===\n");
	int orc = phxfs_open(dev);
	check(orc == 0, "phxfs_open rc=%d", orc);
	if (orc != 0) {
		close(fd);
		printf("\n%d run, %d failed\n", g_run, g_fail);
		return 1;
	}
	check(libcuda_matches(), "libcuda matches iluvatar.ko");
	char bus[32] = {};
	if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) == cudaSuccess)
		printf("  PCI %s\n", bus);

	gpu_regmem(dev);
	printf("\n=== sync read/write ===\n");
	gpu_roundtrip(fd, dev, 4096, 1 * 1024 * 1024, 0x21, "4KiB");
	gpu_roundtrip(fd, dev, 64 * 1024, 2 * 1024 * 1024, 0x22, "64KiB");
	gpu_roundtrip(fd, dev, 1024 * 1024, 4 * 1024 * 1024, 0x23, "1MiB");
	gpu_partial(fd, dev);
	printf("\n=== sync batch ===\n");
	gpu_batch(fd, dev, false, 10 * 1024 * 1024, "sync batch");
	printf("\n=== async batch ===\n");
	gpu_batch(fd, dev, true, 12 * 1024 * 1024, "async batch");
	gpu_async_pipeline(fd, dev);
	gpu_cancel_and_destroy(fd, dev);
	gpu_stream(fd, dev);
	gpu_large_chunk(fd, dev);

	phxfs_close(dev);
	close(fd);
	printf("\n%d run, %d failed\n", g_run, g_fail);
	return g_fail ? 1 : 0;
}
