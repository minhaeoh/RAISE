import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 데이터
labels = ['Not relevant at all   ', 'Superficially relevant', 'Partially relevant   ', 'Fully relevant']
label_colors = ['#b2182b', '#e6550d', '#2166ac', '#1a9850']
colors = ['#2166ac','#e6550d','#b2182b','#1a9850']
explode = (0, 0, 0, 0)
data_list = [
    [37.17,12.48,37.71,12.65],
    [36.85,15.77,35.75,11.64],
    [35.57,9.95,38.44,16.04],
    [45.45,13.28,25.74,15.53]
]
captions = [ 'RAG', 'Step-back', 'Hyde','RAISE']  # 아래에 쓸 제목

fig, axs = plt.subplots(2, 2, figsize=(12,12))

# 범례용 빈 축
axs = axs.flatten()

# 범례용 빈 축 제거
for ax in axs:
    ax.axis('equal')  # 원형 유지

# 파이 차트 그리기
for i in range(4):
    wedges, texts, autotexts = axs[i].pie(
        data_list[i],
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        startangle=120,
        textprops={'fontsize': 23, 'fontweight': 'bold'}
    )
    axs[i].text(0, -1.145, captions[i], ha='center', fontsize=26, fontweight='bold')

# 범례 (하단 중앙에 한 줄로)
legend_patches = [Patch(facecolor=label_colors[i], label=labels[i]) for i in range(len(labels))]
fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.014),
           ncol=2, frameon=False, prop={'size': 24, 'weight': 'bold'})

plt.tight_layout()
plt.subplots_adjust(left=0.001, right=0.999, bottom=0.095, top=0.999,hspace=0.005)
plt.savefig("chart_2x2.pdf", dpi=300)
plt.show()
