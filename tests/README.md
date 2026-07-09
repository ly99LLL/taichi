# tests 自动测试

跑 `python -m pytest tests/ -v` 验证代码没坏。

| 文件 | 测什么 |
|------|--------|
| `__init__.py` | 测试包标记 |
| `test_config.py` | 所有常数值在合理范围 |
| `test_motion.py` | 运动分析：手部检测、速度计算、迟滞 |
| `test_physics.py` | 粒子系统：初始化、单手/双手更新、GPU kernel 编译 |
