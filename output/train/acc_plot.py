from cProfile import label
from threading import local
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/cifar10-vit_7_4_32-32-origin/summary.csv')
data2 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/cifar10-vit_7_4_32-32-addDW/summary.csv')
data3 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/cifar10-vit_7_4_32-32-weightSharingDWConv/summary.csv')
data4 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/cifar10-vit_7_4_32-32-weightSharingDWConv(h=16)/summary.csv')
data5 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/cifar10-vit_7_4_32-32-weightSharingDWConv(all_channels)/summary.csv')
data6 = pd.read_csv('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/20220906-124023-vit_7_4_32-32/summary.csv')

plt.figure(figsize=(12, 8))

# plt.plot(data1['eval_top1'], label='Original ViT (baseline)')
# plt.plot(data2['eval_top1'], label='Directly add DWConv')
# plt.plot(data3['eval_top1'], label='Add DWConv via a trainable matrix (bmm)')

plt.plot(data1['eval_top1'], label='Original ViT (vit_7_4_32-32, #heads=4)')
plt.plot(data2['eval_top1'], label='ViT with additional DWConv (#heads=4)')
plt.plot(data3['eval_top1'], label='ViT with additional weight-sharing DWConv for each head (#heads=16)')
plt.plot(data4['eval_top1'], label='ViT with additional weight-sharing DWConv for each head (#heads=4)')
plt.plot(data5['eval_top1'], label='ViT with additional weight-sharing DWConv for all channels (#heads=4)')
plt.plot(data6['eval_top1'], label='Original ViT (vit_7_4_32-32, #heads=16)')

# plt.plot(data1['eval_top5'], label='Original Compact ViT-top5')
# plt.plot(data2['eval_top5'], label='Add DWConv after attention layer-top5')
# plt.plot(data3['eval_top5'], label='Add DWConv synchronized with attention layer-top5')

plt.xlabel('Epoch')
plt.ylabel('Acc-top1 (%)')
plt.title('Accuracy-top1 of different models')
plt.legend(loc='lower right')
plt.show()
plt.savefig('/home/wenxuanzeng/projects/Compact-Transformers-main/output/train/vit-accuracy-top1.png')
