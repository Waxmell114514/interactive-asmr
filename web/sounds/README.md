# 自定义音频素材

默认的环境音（雨声、壁炉）和音效（翻书、倒茶、布料摩擦）都在浏览器里程序化生成，不需要任何素材文件。

想换成真实录音，把文件放到这里，文件名与名称一致即可，支持 `.ogg` / `.mp3` / `.wav`：

- `ambience/rain.ogg`、`ambience/fire.ogg`：立体声、可无缝循环
- `sfx/page_turn.wav`、`sfx/tea_pour.wav`、`sfx/fabric_rustle.wav`：单声道（会跟着角色位置做 HRTF 定位）

新增音效名时，还要把它加进后端白名单 `asmr/tags.py` 的 `DEFAULT_SFX`，否则 LLM 输出的 `[sfx: xxx]` 会被丢弃。
请只使用你有权使用的素材。
