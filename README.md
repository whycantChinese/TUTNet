这是使用前，一些作者想要告诉你的事情，如果有耐心，希望你能看完，包含无法在论文原文中说明的一些细节，<span style="color:red">文末最后有网络引入的代码供你使用 </span>

这是贴主截止硕士毕业的唯一一篇一作小论文，在很多数据集上的指标并不算优秀，训练和检验框架请参考UDTransNet那个文章，我觉得那个文章的代码框架是不错的（利益无关）

<span style="color:#ff0000">如果需要我的权重请评论区联系我</span>，或者有什么问题或者质疑或者需要代码片段都可以联系我，我觉得无所谓，但**不想公开，可私下获取，可免费远程指导复现

文章不是很好的文章，但是代码是可用可验证的，这点可以保证，实际实验中采取了很多不同型号的卡去实验，上下浮动绝对差基本不高于0.5%，过于小的数据集大约浮动1%，文章全部采取的原图数据集，没有经过任何缩放处理的，如果你用了那种处理过数据集，可能排名会变化，精准度也会提高一些

我是一名坚定的传统深度学习学生，我认为实验需要划分验证集和测试集，并且坚决反对直接取测试集上最好的结果，因此文中部分引用网络的实验结果显著低于他们原来的论文，请你放心，我绝对不会故意踩别人的文章和论文，有一些过于表现差劲的文章被我剔除了，比如一些Mamba系的网络

文章当中有个细节有点问题，有一个表格，我的表现并不是第二，但是我依然给了蓝色，这个我当时没有注意到，是一个错误

下面是我网络代码，以及导入的过程，如果有什么地方失败了可以直接问我

<span style="color:red">如果有什么问题，可以直接中文向我提问，因为我是中国人，国人之间用英文交流浪费沟通成本

But if you're a foreigner, you can communicate with me in English, or in any other language that AI understands.</span>

希望你能做出更有意义的科研

# TUTNet

torch >= 2.0.0
cuda >= 11.8

from MYNetpro import MYNet<br>
from TF_configspro import get_model_config<br>

config_vit = get_model_config()<br>
model = MYNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels, img_size=config.img_size)<br> 
