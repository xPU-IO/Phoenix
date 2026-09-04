/*
 * METAX P2P Backend Header File
*/

#include <linux/scatterlist.h>

/*
* 通知 GPU driver 要访问的 mem 区域，不允许跨进程访问，所以没有进程相关参数
*
* @addr[in]: GPU 虚拟地址
* @size[in]： GPU 虚拟地址长度
* @handle[out]: 由 GPU driver 分配并设置, 表示 addr/size 代表的内存区域, 后续调用其它接口时，使用这个参数表示此内存区域
* @callback[in]: GPU driver 回收内存时，调用此接口，peer 端阻塞直到 IO 完成，内存可以释放，如果长时间 io 不能结束，返回 -ETIMEOUT, 成功返回 0
* @arg[in]: GPU driver 调用 callback 时传的参数
* @return
*   0 success
*   <0 errno code
*/
int metax_p2p_acquire_mem(uint64_t addr, size_t size, void **handle, int (*callback)(void *arg), void *arg);


/*
* pin 住 mem 区域并获取总线地址
*
* @handle[in] - metax_p2p_acquire_mem 获取的句柄
* @sgt[out] - 表示 bus 地址的 sg table，由 GPU driver 设置并分配, 存储侧可以结合 metax_p2p_get_page_size 返回的 page size 转成成
*             定长数组的形式，sg table 里的地址/长度都是 page size 对齐的
* @return
*   0 success
*   <0 errno code
*/
int metax_p2p_get_mem(void *handle, struct sg_table **sgt);


/*
* unpin mem
*
* @handle[in] - metax_p2p_acquire_mem 获取的句柄
* @sgt[in] - metax_p2p_get_mem 获取的 sgt, 可以是 NULL
* @return
*   0 success
*   <0 errno code
*/
int metax_p2p_put_mem(void *handle, struct sg_table *sgt);


/*
* 获取 mem 区域 bus 地址到 cpu 地址的 offset
*
* @handle[in] - metax_p2p_acquire_mem 获取的句柄
* @return
*   bus 地址到 cpu 地址的 offset, bus 地址加上此 offset 可以得到 cpu 地址
*/
uint64_t metax_p2p_get_bus_offset(void *handle);


/*
* 获取 mem 区域的 page size, metax_p2p_get_mem 后调用
*
* @handle[in] - metax_p2p_acquire_mem 获取的句柄
* @return
*   mem 区域的 page size
*/
uint32_t metax_p2p_get_page_size(void *handle);


/*
* 通知 GPU driver mem 区域使用完毕
*
* @handle[in] - metax_p2p_acquire_mem 获取的句柄
* @return
*   0 success
*   <0 errno code
*/
void metax_p2p_release_mem(void *handle);
