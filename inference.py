'''
CLIP能力演示

1、对图片做分类
2、对图片求相图片

'''

from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from clip import CLIP
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=CLIP().to(DEVICE) # 模型
model.load_state_dict(torch.load('model.pth'))

model.eval()    # 预测模式

'''
1、对图片分类
'''
image,label=dataset[0]
print('正确分类:',label)
plt.imshow(image.permute(1,2,0))
plt.show()

targets=torch.arange(0,10)  #10种分类
logits=model(image.unsqueeze(0).to(DEVICE),targets.to(DEVICE)) # 1张图片 vs 10种分类
print(logits)
print('CLIP分类:',logits.argmax(-1).item())

'''
2、图像相似度
'''
other_images=[]
other_labels=[]
for i in range(1,101):
    other_image,other_label=dataset[i]
    other_images.append(other_image)
    other_labels.append(other_label)

# 其他100张图片的向量
other_img_embs=model.img_enc(torch.stack(other_images,dim=0).to(DEVICE))

# 当前图片的向量
img_emb=model.img_enc(image.unsqueeze(0).to(DEVICE))

# 计算当前图片和100张其他图片的相似度
logtis=img_emb@other_img_embs.T
values,indexs=logtis[0].topk(5) # 5个最相似的

plt.figure(figsize=(15,15))
for i,img_idx in enumerate(indexs):
    plt.subplot(1,5,i+1)
    plt.imshow(other_images[img_idx].permute(1,2,0))
    plt.title(other_labels[img_idx])
    plt.axis('off')
plt.show()