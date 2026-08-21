# Tessellation

这个项目用于整理和实验平面镶嵌相关的问题，尤其关注单一形状能否密铺平面、能否强迫非周期，以及如何用程序搜索新的候选形状。

## Hat 和 $\mathrm{Tile}(a,b)$ 族

2023 年发现的 hat 是第一个著名的平面非周期单砖例子。它属于一个连续族 $\mathrm{Tile}(a,b)$。这个族可以理解为在固定角度结构下改变两类边长参数得到的一族多边形；除去 $a=0$、$b=0$ 这两个退化情形，一般成员都是 13 边形。

几个重要特例：

- hat 对应 $\mathrm{Tile}(1,\sqrt{3})$。
- turtle 对应 $\mathrm{Tile}(\sqrt{3},1)$。
- $\mathrm{Tile}(1,1)$ 是特殊的等边情形。

这些对象之间有几种不同的密铺表示，需要区分：

- 普通 $\mathrm{Tile}(a,b)$ 族成员，例如 hat，在允许镜像的意义下强迫非周期密铺。
- $\mathrm{Tile}(1,1)$ 在禁止镜像、只允许平移和旋转时强迫非周期；但如果允许镜像，它可以周期密铺。因此它是弱手性非周期单砖。
- companion substitution 表示使用 $\mathrm{Tile}(a,b)$ 和 $\mathrm{Tile}(b,a)$ 一起拼。默认 $a=1, b=\sqrt{3}$ 时，就是 hat/turtle 表示。它和“一个 hat 加它自己的镜像”不是同一个表示。

当前项目中的相关脚本：

- `tile_one_one.py`：生成 $\mathrm{Tile}(1,1)$ 的同手性替换密铺补丁。
- `tile_ab_companion_substitution.py`：生成 $\mathrm{Tile}(a,b)$ 与 $\mathrm{Tile}(b,a)$ 的 companion substitution 补丁。
- `tile_ab.py`：基于论文中的真实 patch 数据，生成 hat 与其镜像块组成的补丁。

## 搜索算法设想

目标是搜索新的可镶嵌候选形状，尤其是可能强迫非周期的单一形状。一个设想是不直接枚举任意多边形坐标，而是从周期性点集或周期性 cell graph 出发。

基本思路：

1. 选择一个周期性点集或周期性 cell graph。
2. 每个点对应一个 Voronoi 单元；如果点集中存在多种局部处境，就可能有多种单元类型（不用真的计算 Voronoi 图，只在点阵层面尝试密铺）。
3. 在离散图层面枚举由 `k` 个相邻 cell 构成的连通 polyform。
4. 对枚举出的 polyform 做去重。
5. 将每个 polyform 当作一个候选单砖，尝试在有限区域上做 exact cover、回溯或 SAT 求解。
6. 对能铺较大有限区域的候选，继续搜索周期铺法；若找到周期铺法，则淘汰。
7. 对剩余候选，计算真实 Voronoi 多边形并合并对应 cell，导出图片供人工筛选。

torus exact cover 可用来搜索周期铺法：把周期图按两个独立平移向量取有限商，也就是把一个超胞的相对边周期性粘起来，然后在这个 torus 上做 exact cover。实现时可用按面积从小到大枚举。若找到解，则候选存在周期密铺，淘汰。
