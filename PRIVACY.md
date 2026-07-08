# Privacy

本项目可能处理摄像头画面和本地太极视频。为了保护隐私，公开仓库只保留可运行源码、公开文档和姿态模板。

## 不应提交的内容

- 原始视频、参考视频、渲染成片和屏幕录制
- 音频文件、逐帧截图、调试帧和临时渲染目录
- `.env`、密钥、证书、令牌、私钥和账号配置
- 本地助手记录、工作过程记录或包含私人素材文件名的笔记

## 已设置的保护

`.gitignore` 已排除常见视频、音频、临时输出、私有目录、凭证文件、`CLAUDE.md`、`agent.md` 和 `jimeng-*.png` 源图。

## 提交前隐私检查

```powershell
git status --short
git ls-files | rg -i "\.(mp4|mov|avi|mkv|webm|m4v|wav|mp3|aac|m4a)$"
rg -n -uu -i "api[_-]?key|secret|token|password|authorization|bearer|private key|client_secret" .
```

视频或音频检查应无输出。敏感词扫描如有命中，需要逐条确认是否只是文档说明。

## 运行时说明

摄像头模式和视频渲染都在本机处理，不会主动上传画面或视频。只有当你手动执行 `git push` 时，已被 Git 跟踪的文件才会发布到远程仓库。
