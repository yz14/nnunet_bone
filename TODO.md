# 核心原则

## 质量第一
- 宁可多花时间，也要保证代码质量
- 充分思考、分析后再动手实现
- 不要为了快速完成而牺牲代码质量

## 分步完成
- 如果当前对话无法完成所有功能，主动拆分为多轮对话
- 每轮只专注完成一个清晰的目标
- 不贪多，确保每一步都高质量完成

## 充分调研
- 如有需要，充分、彻底地搜索和调研
- 分析和掌握现有的高质量功能实现和算法
- 借鉴业界最佳实践，不要闭门造车

## 调试支持
- 如有需要，可以加入 debug/logging 函数辅助开发
- 通过日志输出帮助定位和解决问题
- 调试代码可在功能稳定后标注或移除

## 代码质量  
- 注意代码尽可能模块化设计，职责尽可能的分离，不要把所有代码写在一个文件里，不方便后续理解和维护  
- 注意代码的复用性，不要写重复的代码  

## 沟通规范
- **开始前**：说明你理解的任务目标和将遵守的规则
- **进行中**：如需拆分，明确告知本轮将完成什么
- **完成后**：总结本轮成果，说明后续计划（如有）

测试环境为：**py310**

这个项目是利用nnUnet进行分割的代码。  

## TODO  
1. 代码放到服务器运行时，处理数据阶段python scripts/01_prepare_data.py --config configs/default.yaml就被killed了，是代码问题吗？需要仔细，彻底，细致的检查全流程的代码。  
2026-04-13 17:50:58 [INFO] src.data_prep —   [label] s0921-seg.nii.gz → s0921.nii.gz (remapping labels)
2026-04-13 17:50:58 [INFO] src.data_prep —     Same voxel grid — saving with CT affine, no resampling.
2026-04-13 17:50:58 [INFO] src.data_prep —   [train] s0922.nii.gz → s0922_0000.nii.gz
2026-04-13 17:50:58 [INFO] src.data_prep —   [label] s0922-seg.nii.gz → s0922.nii.gz (remapping labels)
2026-04-13 17:50:58 [INFO] src.data_prep —     Same voxel grid — saving with CT affine, no resampling.
2026-04-13 17:50:58 [INFO] src.data_prep —   [train] s0923.nii.gz → s0923_0000.nii.gz
2026-04-13 17:50:58 [INFO] src.data_prep —   [label] s0923-seg.nii.gz → s0923.nii.gz (remapping labels)
2026-04-13 17:50:58 [INFO] src.data_prep —     Same voxel grid — saving with CT affine, no resampling.
2026-04-13 17:50:59 [INFO] src.data_prep —   [train] s0924.nii.gz → s0924_0000.nii.gz
2026-04-13 17:50:59 [INFO] src.data_prep —   [label] s0924-seg.nii.gz → s0924.nii.gz (remapping labels)
2026-04-13 17:51:00 [INFO] src.data_prep —     Same voxel grid — saving with CT affine, no resampling.
2026-04-13 17:51:00 [INFO] src.data_prep —   [train] s0925.nii.gz → s0925_0000.nii.gz
2026-04-13 17:51:00 [INFO] src.data_prep —   [label] s0925-seg.nii.gz → s0925.nii.gz (remapping labels)
2026-04-13 17:51:00 [INFO] src.data_prep —     Same voxel grid — saving with CT affine, no resampling.
2026-04-13 17:51:00 [INFO] src.data_prep —   [train] s0927.nii.gz → s0927_0000.nii.gz
2026-04-13 17:51:00 [INFO] src.data_prep —   [label] s0927-seg.nii.gz → s0927.nii.gz (remapping labels)
Killed
(py310) yzhen@imedway:/data0/yzhen/projects/BoneSeg$