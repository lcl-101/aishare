# Youtube 节目：
## 微软又出王炸！AI多人对话神器 VibeVoice 本地部署，独家修复官方 WebUI 关键 Bug！
## https://youtu.be/NdTSC3gDx1k

# 安装指南
## 创建运行环境
conda create -n VibeVoice python=3.10 -y    
conda activate VibeVoice  

## 克隆文件
git clone https://github.com/microsoft/VibeVoice.git  
cd VibeVoice/  

## 安装依赖组件
pip install -e .  
pip install flash-attn --no-build-isolation  

## 准备模型文件
huggingface-cli download microsoft/VibeVoice-1.5B --local-dir checkpoints/VibeVoice-1.5B  
huggingface-cli download WestZhang/VibeVoice-Large-pt --local-dir checkpoints/VibeVoice-7B  


cp -r /export/demo-softice/models/VibeVoice/ checkpoints  

## 推理
python app.py  


## 示例 

Speaker 1: 哎，你这周看 Google 那个 Gemini 的发布视频了吗？我看完人都傻了，感觉像在看科幻片。  
Speaker 2: 看了看了，就是那个能实时识别手绘，还能跟你互动的视频吧？确实，第一眼看上去非常惊艳。  
Speaker 1: 可不是嘛！你画个鸭子，它能认出来，你再给鸭子加个海浪，它马上就能联想到“冲浪的鸭子”。这哪是 AI，这简直就是个有“灵性”的小伙伴啊。感觉 Google 这次是真把压箱底的东西掏出来了。  
Speaker 2: 嗯，是下了血本了。不过后来不是有人说，那个 Demo 视频为了效果，经过了剪辑和后期处理，并不是真正的实时互动录像。  
Speaker 1: 哎，你总是这么一针见血。是，Google 后来也承认了，说视频是为了展示 Gemini 的多模态能力。但我觉得重点不在这，重点是它证明了 AI 真的可以“原生”地同时理解图像、声音和文字了。  
Speaker 2: “原生多模态”，这个词是关键。跟现在的 GPT-4 有什么本质区别吗？帮大家解释解释。  
Speaker 1: 简单来说，GPT-4 更像是一个语言天才，后面通过学习才学会了看图。而 Gemini 从“出生”开始，就是同时用文字、图片、音频这些数据一起喂大的，它天生就能通感，理解力会更融会贯通。  
Speaker 2: 我明白了，一个像是后天学会的多才多艺，一个则是天生的全能选手。这么说，这是 Google 对 OpenAI 的一次正面总攻啊。  
Speaker 1: 没错！而且你看它还分了三个版本，最强的 Ultra 对标 GPT-4，中杯 Pro 已经放进 Bard 里了，小杯 Nano 甚至能直接在手机上跑。这是要海陆空全面开战了。  
Speaker 2: 所以对我们普通人来说，最直接的影响就是，以后手机上的 Google 助手，或者电脑上的 Google 搜索，会变得超级聪明？  
Speaker 1: 我觉得这绝对是趋势。以后搜索可能不再是给你一堆链接，而是直接给你一个整合好的、图文并茂的答案，甚至帮你规划好行程、写好邮件。这才是真正的“个人助理”。  
Speaker 2: 听起来很美好，但挑战也不小吧。技术真正落地到每个产品上的体验如何，还有成本和安全问题。AI 越强，犯错的时候可能也越离谱。  
Speaker 1: 那是肯定的。但不管怎么说，巨头开始“卷”起来，对用户总是好事。我们就搬好小板凳，看这场 AI 大战的好戏吧。  
Speaker 2: 哈哈，没错。坐等我们的数字生活被再一次颠覆。  




  












 
















