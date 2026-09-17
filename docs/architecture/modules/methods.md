# 可选择的光传输方法

训练与评估只从 `methods` 导入 `build_transport`；渲染器只使用统一 forward 接口。
`--representation` 及 checkpoint 的同名配置字段确定方法，默认 `directional_port_v1`。

## 模块边界

- `methods/base.py`：`TransportBase`最小接口、`PortTransport`公共端口实现、源/接收光查询类型、方向编码、空间分配、
  质量测度、点光源照明、直接光着色和公共forward。
- `methods/anchor.py`：原 `learned_anchor_exchange`。
- `methods/directional.py`：原 `directional_port_v1` 的方向池化与读取。
- `methods/paired_port.py` / `methods/local_frame.py`：两项研究候选，见[实现说明](research_methods.md)。
- `methods/__init__.py`：显式方法注册表、配置补全、命令行参数及工厂。
- `renderer.py`：属性光栅化、期望深度接收点、阴影与alpha合成。
- `train.py` / `evaluate.py`：通用优化、数据协议、观察变换、checkpoint与指标。

原根目录 `transport.py` 与 `directional_transport.py` 已迁入包，重复直接光计算合并。
参数的state_dict名称保持不变，既有anchor/directional checkpoint可严格加载。
旧checkpoint缺少representation时仍按原合同解释为learned_anchor_exchange。
`--init-checkpoint`仅用于同方法同尺寸的权重初始化，重新创建optimizer；跨方法使用
`--init-geometry`，仍需满足训练划分一致性要求。

## 公共接口

```python
forward(gaussians, receivers, eye, light_pos, light_intensity,
        source_visibility, port_active=True) -> Tensor[M, 3]
```

返回线性前景RGB。方法不得再次做alpha合成、背景、gamma、裁剪或测试标定拟合。
`port_active=False`返回直接光。公共forward生成两个查询：

- `SourceLight`：归一化位置xyz、features、incident、指向光源的direction、mass。
- `ReceiverLight`：归一化位置xyz、features、incident、指向相机的direction、直接光response。

继承PortTransport的方法实现 `exchange_radiance(source, receiver)`，返回已经组合直接光和非局部光的RGB。
材质变化可覆盖 `material_directions` / `direct_response`；方向端口变化可继承
`DirectionalTransport`并覆盖接收空间查询或exchange函数。
若新方法不使用端口，直接继承TransportBase并实现forward，无需实现exchange hook或创建端口参数。
构造参数仍需包含feature_dim，并调用基类初始化light_scale；这两项属于训练器和checkpoint合同。

## 添加方法

1. 在methods中新增一个实现模块，继承合适的基类，直接替换所需hook；不要复制训练器。
2. 声明 `defaults`（全部需保存的构造参数）与 `cli_fields`（允许命令行配置的子集）。
   构造函数接收这些参数以及light_scale，所有可学习模块注册为nn.Module/nn.Parameter。
3. 在 `methods/__init__.py` 导入该类，并在 `METHODS` 增加唯一名称。
4. `train.py --representation NAME --help`自动展示对应参数；训练写出完整配置，
   评估从checkpoint恢复方法和参数，网络optimizer自动包含新增参数。
5. 添加与方法性质相关的公式、梯度、初始化退化和checkpoint测试。

六场景队列沿用 `configs/validation.json`：修改 `train.representation`及实验name，
补充所选方法参数，删去不属于该方法的参数。每个job记录实际representation，
源码归档递归包含methods包；历史实验输出使用原配置与源码。

## 重构验证

与重构前Git提交2204cdc直接比较，anchor与directional两种方法在相同权重和输入下：
开/关端口的CPU float64输出差均为0；参数梯度匹配；state_dict键一致。
记录：`runs/method_refactor_20260917/refactor_parity.json`。
