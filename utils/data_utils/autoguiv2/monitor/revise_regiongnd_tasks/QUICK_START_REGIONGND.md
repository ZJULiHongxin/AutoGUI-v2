# 🚀 FuncRegionGnd 任务修订工具 - 快速开始

## ✅ 当前状态

服务已成功启动并运行在端口 **17806**！

---

## 📊 测试结果

- ✅ **服务状态**: 正在运行 (PID: 3599939)
- ✅ **API端点**: 全部正常
- ✅ **前端页面**: 可访问
- ✅ **数据集数量**: 5个 (osworld_g, screenspot_pro, agentnet, amex, androidcontrol)
- ✅ **样本总数**: 268+ (仅osworld_g)

---

## 🌐 访问地址

### 如果你在服务器本地
```
http://localhost:17806
```

### 如果需要远程访问
```
http://<服务器IP>:17806
```

> **注意**: 服务已绑定到 `0.0.0.0`，可以从任何网络访问（如果防火墙允许）

---

## 🎯 主要功能

### 1️⃣ 浏览样本
- 选择数据集 (Dataset选择器)
- 选择样本 (Sample选择器)
- 使用 Previous/Next 按钮快速导航

### 2️⃣ 修改数据
- **问题文本**: 直接在文本框中编辑
- **正确答案**: 点击 A/B/C/D 按钮
- **BBox坐标**: 点击 "Fix BBox" 使用交互式工具

### 3️⃣ 保存修改
- 自动保存：切换样本时自动保存
- 智能检测：如果没有修改，不会创建修订记录

### 4️⃣ 查看选项
- 所有选项（A/B/C/D）会显示在 "Options" 区域
- 包含 region_type, description, bbox 等信息
- 正确答案会高亮显示

---

## 🔧 管理命令

### 查看服务状态
```bash
ps aux | grep revise_regiongnd_tasks
```

### 停止服务
```bash
pkill -f revise_regiongnd_tasks
```

### 重启服务
```bash
cd /mnt/nvme0n1p1/hongxin_li/highres_autogui
./utils/data_utils/autoguiv2/monitor/run_regiongnd_ui.sh
```

### 查看日志
```bash
# 服务在前台运行时会直接显示日志
# 如果在后台运行，可以这样查看：
tail -f /tmp/regiongnd_reviser.log  # (如果配置了日志文件)
```

---

## 📁 数据文件位置

### 原始数据
```
/mnt/vdb1/hongxin_li/AutoGUIv2/{dataset}/FuncRegion/grounding_mode/*_result.json
```

### 修订数据（自动创建）
```
/mnt/vdb1/hongxin_li/AutoGUIv2/{dataset}/FuncRegion/grounding_questions_corrections.json
```

---

## 🎨 UI 功能说明

### 样本状态标识
- 🆕 **Untouched**: 未修改
- ✏️ **Modified**: 已修改
- 🗑️ **Abandoned**: 已废弃

### BBox 修正工具
1. 点击 "Fix BBox" 按钮
2. 点击 "Select Top-Left"，然后在图像上点击左上角
3. 点击 "Select Bottom-Right"，然后在图像上点击右下角
4. 或直接在输入框中输入坐标

**高级功能**（需要OpenCV.js加载完成）:
- **长按（500ms）**: 自动吸附到边缘
- **Ctrl + 长按**: 自动吸附到角点

---

## 📊 API 端点参考

所有API端点的基础URL: `http://localhost:17806`

| 端点 | 方法 | 说明 | 示例 |
|------|------|------|------|
| `/health` | GET | 健康检查 | `curl http://localhost:17806/health` |
| `/api/datasets` | GET | 获取数据集列表 | `curl http://localhost:17806/api/datasets` |
| `/api/images?dataset=xxx` | GET | 获取样本列表 | `curl "http://localhost:17806/api/images?dataset=osworld_g"` |
| `/api/sample?...` | GET | 获取样本详情 | `curl "http://localhost:17806/api/sample?dataset=osworld_g&json_file=xxx.json&q_idx=0"` |
| `/api/save_correction` | POST | 保存修订 | (通过前端表单提交) |

---

## 🔄 与元素级别工具对比

| 特性 | 元素级别 (17805) | 区域级别 (17806) |
|------|-----------------|-----------------|
| 数据源 | FuncElemGnd/ | FuncRegion/grounding_mode/ |
| 修改内容 | action_type问题 | 选择题答案 |
| 答案格式 | 无选项 | A/B/C/D |
| 运行状态 | - | ✅ 正在运行 |

---

## ⚠️ 注意事项

1. **自动保存**: 切换样本时会自动保存当前修改
2. **OpenCV加载**: 首次打开需要等待几秒加载OpenCV.js
3. **并发编辑**: 不支持多人同时编辑同一数据集
4. **数据备份**: 建议定期备份corrections文件

---

## 🐛 故障排查

### 问题: 页面无法访问
**解决**: 
```bash
# 检查服务是否运行
ps aux | grep revise_regiongnd_tasks
# 如果没有运行，重新启动
./run_regiongnd_ui.sh
```

### 问题: No datasets found
**解决**: 
检查数据目录结构是否正确：
```bash
ls -la /mnt/vdb1/hongxin_li/AutoGUIv2/*/FuncRegion/grounding_mode/
```

### 问题: OpenCV is not loaded yet
**解决**: 
等待几秒钟或刷新页面，OpenCV.js需要时间加载

---

## 📞 更多信息

- **测试报告**: `./revise_regiongnd_tasks/TEST_RESULTS.md`
- **详细文档**: `./revise_regiongnd_tasks/README.md`
- **源代码**: `./revise_regiongnd_tasks.py`

---

## ✨ 下一步

现在你可以：
1. 🌐 打开浏览器访问 http://localhost:17806
2. 🔍 选择一个数据集开始浏览
3. ✏️ 尝试修改一个样本
4. 💾 观察自动保存功能

祝使用愉快！🎉

