"""Export frozen experiment evidence as paper figures and table source files.

No training or test-set selection. Small CPU forwards reproduce only the
previously fixed representative/failure images; full test CSVs remain unchanged.
"""
from pathlib import Path
import hashlib
import html
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd
import torch

from data_pipeline import load_test_partition
from models import predict_clean
from noise import fixed_evaluation_noise
from paper_figures import load_model
from metrics import per_image_metrics
from references import gaussian_denoise

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'dist/formal_results_20261022T004235964922Z/formal'
REPORT = WORK / 'report/test'
OUT = ROOT / 'outputs/paper_assets'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
DATA = OUT / 'source_data'
for folder in (FIG, TAB, DATA):
    folder.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

audit = read(ROOT / 'outputs/review/final_test/audit.json')
assert audit['status'] == 'passed'
assert audit['archive_sha256'] == sha(ROOT / 'dist/formal_results_20261022T004235964922Z.zip')
for name in ('mean_std.csv', 'results_by_seed.csv', 'paired_ablation.csv',
             'factorial_effects.csv', 'per_digit_by_seed.csv', 'per_digit_mean_std.csv'):
    shutil.copy2(REPORT / name, DATA / name)
mean = pd.read_csv(REPORT / 'mean_std.csv')
by_seed = pd.read_csv(REPORT / 'results_by_seed.csv')
selection = read(WORK / 'gradient_selection.json')
font_manager.fontManager.addfont('C:/Windows/Fonts/simsun.ttc')
FONT = font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc').get_name()
plt.rcParams.update({'font.family': [FONT, 'DejaVu Sans'], 'font.size': 10.5,
                     'axes.unicode_minus': False, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelsize': 10.5,
                     'legend.fontsize': 9.5, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
COLORS = ['#235789', '#C47426', '#658238', '#9A5577', '#555555']
MARKERS = ['o', 's', '^', 'D', 'x']
SEEDS = [42, 2026, 3407]
registry = []

def save(fig, name, title, section, caption, sources, table=False):
    folder = TAB if table else FIG
    for ext in ('png', 'pdf'):
        fig.savefig(folder / f'{name}.{ext}', dpi=320, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    registry.append({'id': name, 'kind': 'table' if table else 'figure', 'title': title,
                     'section': section, 'caption': caption,
                     'png': f'{folder.name}/{name}.png', 'pdf': f'{folder.name}/{name}.pdf',
                     'sources': sources})
    print('Created', name, flush=True)

def style(ax, xlabel=None, ylabel=None):
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def row(variant, sigma):
    return mean.loc[(mean.variant == variant) & np.isclose(mean.sigma, sigma)].iloc[0]

def fmt(record, metric, digits=3):
    m, sd = record[metric+'_mean'], record[metric+'_sd']
    return f'{m:.{digits}f}' if pd.isna(sd) else f'{m:.{digits}f} ± {sd:.{digits}f}'

def table(name, title, section, columns, rows, caption, widths=None):
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(TAB / f'{name}.csv', index=False, encoding='utf-8-sig')
    text = '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---']*len(columns)) + ' |\n'
    text += ''.join('| ' + ' | '.join(map(str, values)) + ' |\n' for values in rows)
    (TAB / f'{name}.md').write_text(f'{title}\n\n{text}\n{caption}\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7.7, .38*(len(rows)+1)+.25))
    ax.axis('off')
    artist = ax.table(cellText=rows, colLabels=columns, cellLoc='center', colLoc='center',
                      colWidths=widths, bbox=(0, 0, 1, 1))
    artist.auto_set_font_size(False)
    artist.set_fontsize(10)
    for (r, c), cell in artist.get_celld().items():
        cell.set_facecolor('white')
        cell.set_linewidth(0)
        cell.visible_edges = ''
        if r == 0:
            cell.visible_edges = 'TB'
            cell.set_linewidth(.9)
        if r == len(rows):
            cell.visible_edges = 'B'
            cell.set_linewidth(.9)
    fig.tight_layout(pad=.3)
    save(fig, name, title, section, caption, ['mean_std.csv', 'results_by_seed.csv',
         'frozen training configs / gradient_selection.json where applicable'], table=True)

def box(ax, xy, text, w=.19, h=.16, fill='#F4F6F8', size=10):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.008,rounding_size=0.008',
                               linewidth=.8, edgecolor='#475569', facecolor=fill))
    ax.text(x, y, text, ha='center', va='center', fontsize=size)

def arrow(ax, a, b):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'lw': .9, 'color': '#475569'})

def diagrams():
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.set(xlim=(0,1.06), ylim=(0,1)); ax.axis('off')
    for x, text in zip([.17,.5,.83], ['官方训练集\n60,000 张', '固定划分 seed=2026\n训练 55,000\n验证 5,000', '独立测试集\n10,000 张']):
        box(ax,(x,.87),text,w=.27,h=.20)
    arrow(ax,(.31,.87),(.36,.87))
    for x, text in zip([.17,.5,.83], ['验证集选择 λ\n3 次选参后冻结', '动态加噪训练\n19 配置 × 3 种子', '冻结第 20 轮模型\n57 个正式模型']):
        box(ax,(x,.53),text,w=.27,h=.18)
    arrow(ax,(.43,.78),(.17,.63)); arrow(ax,(.5,.78),(.5,.63))
    arrow(ax,(.31,.53),(.36,.53)); arrow(ax,(.64,.53),(.69,.53))
    # This arrow denotes the separate test source, not model selection.
    ax.plot([.975,1.025,1.025],[.87,.87,.15],color='#475569',lw=.9)
    arrow(ax,(1.025,.15),(.97,.15))
    box(ax,(.5,.15),'统一测试：σ = 0.0, 0.1, …, 0.6\n57 模型 + 原始噪声图 / 高斯滤波 → 指标、重建图与消融分析',w=.91,h=.20)
    arrow(ax,(.83,.43),(.83,.26))
    save(fig,'fig01_protocol','数据划分与实验流程','4.1',
         '固定划分用于全部训练种子；λ仅由验证集确定。测试集不参与模型与超参数选择。', ['data_report.json','experiment configs'])

    fig, axes = plt.subplots(2,1,figsize=(7.7,4.5))
    labels = [['784','256','64','z','64','256','784'],
              ['2×28×28','16×14×14','32×7×7','64×7×7','32×14×14','16×28×28','1×28×28']]
    operations = [['Linear','Linear','Linear','Linear','Linear','Linear'],
                  ['Conv\nk3 s2','Conv\nk3 s2','Conv\nk3 s1','TConv\nk4 s2','TConv\nk4 s2','Conv\nk3 s1']]
    for n, ax in enumerate(axes):
        ax.set(xlim=(-.02,1.02),ylim=(0,1)); ax.axis('off')
        ax.text(0,.92, '(a) 全连接自编码器' if n==0 else '(b) 卷积自编码器',fontsize=12)
        xs=np.linspace(.065,.935,7)
        for x,label in zip(xs,labels[n]): box(ax,(x,.55),label,w=.122,h=.22,size=9.5)
        for i in range(6):
            arrow(ax,(xs[i]+.062,.55),(xs[i+1]-.062,.55))
            ax.text((xs[i]+xs[i+1])/2,.24,operations[n][i],ha='center',va='center',fontsize=8.5)
        note = ('z∈{8,16,32,64}；隐藏层 ReLU；输出层 Sigmoid；输出恢复为 1×28×28。' if n==0 else
                '输入为图像与辅助通道；前五层后接 ReLU，末层线性；测试输出截断至 [0,1]。')
        ax.text(.5,.01,note,ha='center',fontsize=9)
    fig.tight_layout(h_pad=1)
    save(fig,'fig02_architecture','全连接与卷积自编码器结构','3.2',
         '卷积张量标注不含批次维。辅助通道在普通模型中为0，在条件模型中为已知σ；c16是基础通道数，非16维瓶颈。卷积核与步长见图，填充均为1。', ['src/models.py'])

    fig,ax=plt.subplots(figsize=(7.5,4.2)); ax.axis('off'); ax.set(xlim=(0,1),ylim=(0,1))
    ax.text(.02,.94,'同一输入与重建目标',fontsize=12)
    ax.text(.5,.81,r'$y=\mathrm{clip}(x+\sigma\epsilon,0,1),\quad \epsilon\sim\mathcal{N}(0,I)$',ha='center',fontsize=13)
    box(ax,(.24,.57),'直接重建\n'+r'$\hat{x}=f_\theta(y,a)$',w=.40,h=.23)
    box(ax,(.76,.57),'残差预测\n'+r'$\hat{x}=y-f_\theta(y,a)$',w=.40,h=.23)
    ax.text(.5,.36,'辅助通道 a：普通模型为 0，条件模型为 σ；两者参数量相同。',ha='center',fontsize=10)
    ax.text(.5,.22,r'$\mathcal{L}=\mathrm{MSE}(\hat{x},x)+\lambda\,\mathcal{L}_{grad}$',ha='center',fontsize=13)
    ax.text(.5,.08,'梯度项：水平与垂直相邻差分的 L1 误差取平均；λ = 0 或 0.1。',ha='center',fontsize=10)
    save(fig,'fig03_variants','重建方式与梯度约束','3.3–3.4',
         '普通/条件输入与直接/残差预测为方法示意，不表示已训练全部组合。梯度损失在未截断重建上计算，评估时统一截断。残差目标对应截断后的有效噪声 y−x。',['src/models.py','src/losses.py','src/noise.py'])

def curves():
    families = [('fc_baseline','FC z16','z16'), ('cae_baseline','CAE','c16'),
                ('cae_gradient','CAE+G','c16'), ('cae_residual','CAE+R','c16'),
                ('cae_residual_gradient','CAE+R+G','c16')]
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.6),sharex=True)
    records=[]
    for r,sigma in enumerate([.3,.5]):
        for k,(prefix,label,suffix) in enumerate(families):
            frames=[]
            for seed in SEEDS:
                job=f'{prefix}_s{round(sigma*10):02d}_{suffix}_seed{seed}'
                f=pd.read_csv(WORK/'runs'/job/'metrics.csv')
                assert len(f)==20
                frames.append(f)
                records.append(f.assign(job_id=job,seed=seed,sigma=sigma))
            for c,metric in enumerate(['train_mse','val_mse']):
                array=np.stack([f[metric] for f in frames]); m=array.mean(0); sd=array.std(0,ddof=1)
                ax=axes[r,c]; x=frames[0].epoch
                ax.plot(x,m,color=COLORS[k],label=label,lw=1.3,linestyle=['-','-','--','-.',':'][k])
                ax.fill_between(x,np.maximum(m-sd,1e-6),m+sd,color=COLORS[k],alpha=.10)
                ax.set_yscale('log'); ax.set_title(f'σ={sigma} '+('训练集' if c==0 else '验证集'))
                ax.set_yticks([.003,.01,.03,.1,.3])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f'{v:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                style(ax,'训练轮数' if r==1 else None,'MSE（对数轴）'); ax.set_xticks([1,5,10,15,20])
        for ax in axes[r]: ax.set_ylim(.003,.3)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='lower center',frameon=False)
    fig.tight_layout(rect=(0,.06,1,1))
    pd.concat(records).to_csv(DATA/'learning_curves.csv',index=False)
    save(fig,'fig05_learning','训练与验证重建误差曲线','5.1.1',
         '曲线为三个训练种子的均值，阴影为样本标准差。显示MSE而非带梯度项的总损失；训练图像动态加噪，验证噪声固定，因此两者不是逐图配对误差。R为残差预测，G为梯度约束。',['runs/*/metrics.csv'])

def numerical_figures():
    fig,axes=plt.subplots(1,2,figsize=(7.5,3.3))
    for k,sigma in enumerate([.3,.5]):
        part=pd.DataFrame([row(f'fc_baseline_s{round(sigma*10):02d}_z{z}',sigma) for z in [8,16,32,64]])
        for ax,metric in zip(axes,['PSNR','SSIM']):
            ax.errorbar([8,16,32,64],part[metric+'_mean'],yerr=part[metric+'_sd'],color=COLORS[k],
                         marker=MARKERS[k],capsize=3,label=f'训练/测试 σ={sigma}',lw=1.2)
            style(ax,'瓶颈维度 z','PSNR / dB' if metric=='PSNR' else metric); ax.set_xticks([8,16,32,64])
    axes[1].legend(frameon=False,fontsize=9)
    fig.tight_layout()
    save(fig,'fig06_bottleneck','瓶颈维度与重建性能','5.2',
         '每点为三个训练种子的均值，误差棒为样本标准差。仅比较FC模型，训练和测试噪声匹配。',['mean_std.csv'])

    fig,axes=plt.subplots(1,2,figsize=(7.5,3.7))
    variants=[('cae_baseline_s03_c16','固定 σ=0.3'),('cae_baseline_s05_c16','固定 σ=0.5'),
              ('cae_mixed_c16','混合噪声'),('cae_conditioned_mixed_c16','混合+已知σ')]
    for k,(variant,label) in enumerate(variants):
        part=mean.loc[(mean.variant==variant)&(mean.sigma>0)].sort_values('sigma')
        assert len(part)==6
        for ax,metric in zip(axes,['PSNR','SSIM']):
            x=part.sigma.to_numpy(); y=part[metric+'_mean'].to_numpy(); sd=part[metric+'_sd'].to_numpy()
            ax.plot(x,y,label=label,color=COLORS[k],marker=MARKERS[k],lw=1.2,ms=4)
            ax.fill_between(x,y-sd,y+sd,color=COLORS[k],alpha=.12)
            style(ax,'测试噪声 σ','PSNR / dB' if metric=='PSNR' else metric)
            ax.set_xticks([.1,.2,.3,.4,.5,.6])
    for ax in axes: ax.axvline(.5,color='#666666',ls=':',lw=.8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.15,1,1))
    save(fig,'fig07_robustness','固定与混合噪声训练的跨噪声表现','5.3',
         '均值及阴影分别表示三种子均值和样本标准差。混合训练σ∈[0.1,0.5)，σ=0.6属于范围外测试；虚线标出0.5边界。条件模型使用已知真实σ。σ=0无噪声诊断完整数据见附录。',['mean_std.csv'])

    paired=pd.read_csv(REPORT/'paired_ablation.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.6,3.5))
    for k,sigma in enumerate([.3,.5]):
        for ax,metric in zip(axes,['PSNR','SSIM']):
            groups=[paired.loc[np.isclose(paired.sigma,sigma)&(paired.variant==v),metric+'_delta'] for v in 'ABCD']
            ax.errorbar(np.arange(4)+(k-.5)*.12,[g.mean() for g in groups],yerr=[g.std(ddof=1) for g in groups],
                        color=COLORS[k],fmt=MARKERS[k],capsize=3,label=f'σ={sigma}')
            ax.axhline(0,color='#777777',ls='--',lw=.8)
            ax.set_xticks(range(4),['A\n基础','B\n+R','C\n+G','D\n+R+G'])
            style(ax,None,'ΔPSNR / dB' if metric=='PSNR' else 'ΔSSIM')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig,'fig08_ablation','残差与梯度约束的配对消融','5.4',
         '同一训练种子内相对基础CAE作差，再汇总三种子均值±样本标准差；不是置信区间。R为残差预测，G为梯度约束。完整MSE、梯度误差及交互效应见表和附录。',['paired_ablation.csv'])

    digits=pd.read_csv(REPORT/'per_digit_by_seed.csv')
    fig,axes=plt.subplots(2,1,figsize=(7.4,4.8),layout='constrained')
    names=[('fc_baseline','FC z16','z16'),('cae_baseline','CAE','c16'),('cae_gradient','CAE+G','c16'),
           ('cae_residual','CAE+R','c16'),('cae_residual_gradient','CAE+R+G','c16')]
    for ax,sigma in zip(axes,[.3,.5]):
        matrix=[]
        for prefix,_,suffix in names:
            part=digits.loc[(digits.variant==f'{prefix}_s{round(sigma*10):02d}_{suffix}')&np.isclose(digits.sigma,sigma)]
            matrix.append(part.groupby('label').SSIM.mean().reindex(range(10)).values)
        matrix=np.array(matrix)
        im=ax.imshow(matrix,vmin=.60,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_yticks(range(5),[n[1] for n in names]); ax.set_xticks(range(10)); ax.set_xlabel('数字类别')
        ax.set_title(f'训练/测试 σ={sigma}')
        for r in range(5):
            for c in range(10): ax.text(c,r,f'{matrix[r,c]:.2f}',ha='center',va='center',fontsize=9,color='white' if matrix[r,c]>.86 else 'black')
    fig.colorbar(im,ax=axes,shrink=.85,label='类别内平均 SSIM')
    save(fig,'fig12_digits','分数字类别的重建表现','附录B',
         '每格先在该数字类别内平均，再对三个训练种子平均。两面板使用同一色标；数值保留两位小数便于阅读，精确结果在附录CSV。',['per_digit_by_seed.csv'])

def image_grid(rows, labels, ids, digits, name, title, caption, error=False):
    count=len(ids)
    fig=plt.figure(figsize=(7.6, len(rows)*.67+.45))
    # Separate label column guarantees long Chinese labels never intersect pixels.
    grid=fig.add_gridspec(len(rows),count+1,width_ratios=[1.5]+[1]*count,
                         left=.015,right=.97,bottom=.09 if error else .02,top=.95,wspace=.07,hspace=.12)
    image_axes=[]
    for r,values in enumerate(rows):
        label_ax=fig.add_subplot(grid[r,0]); label_ax.axis('off')
        label_ax.text(.97,.5,labels[r],ha='right',va='center',fontsize=9.5)
        for c,value in enumerate(values):
            ax=fig.add_subplot(grid[r,c+1]); image_axes.append(ax)
            im=ax.imshow(np.asarray(value).squeeze(),cmap='magma' if error and r==len(rows)-1 else 'gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if r==0: ax.set_title(f'{digits[c]}\nID {ids[c]}',fontsize=8.5,pad=3)
    if error:
        cax=fig.add_axes([.35,.025,.45,.018]);bar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,.25,.5,.75,1])
        bar.set_label('绝对像素误差 |重建 - 原图|',fontsize=9,labelpad=1)
        bar.ax.tick_params(labelsize=8,pad=1)
    save(fig,name,title,'5.5' if name!='fig04_noise' else '4.1',caption,
         ['fixed sample IDs in report/test/*selection.json','runs/*/last.pt','MNIST raw test pixels'])

@torch.no_grad()
def examples():
    torch.set_num_threads(2)
    dataset=load_test_partition(ROOT)
    ids=read(REPORT/'representative_selection.json')['image_ids_by_digit']
    clean=torch.stack([dataset[i]['clean'] for i in ids])
    noisy={s:fixed_evaluation_noise(clean,torch.tensor(ids),s,seed=19300) for s in [.3,.5]}
    image_grid([clean[:,0],noisy[.3][:,0],noisy[.5][:,0]],['原图','σ=0.3','σ=0.5'],ids,list(range(10)),
               'fig04_noise','固定测试样本与高斯加噪','按测试集原始顺序选择每个数字首次出现的图像；同一图像不同σ共用标准正态噪声并按σ缩放，最后截断至[0,1]。')
    filters=read(WORK/'gaussian_selection.json')
    checks=[]
    cache={}
    def infer(job, x, target, selected_ids, sigma):
        if job not in cache: cache[job]=load_model(WORK/'runs'/job,'cpu')
        result=predict_clean(cache[job],x,sigma).clamp(0,1).cpu()
        values=per_image_metrics(target,result)
        saved=pd.read_csv(WORK/'evaluation/test'/job/'per_image.csv.gz')
        saved=saved.loc[np.isclose(saved.sigma,sigma)].set_index('image_id').loc[selected_ids]
        diffs={}
        for metric in ['MSE','PSNR','SSIM']:
            diffs[metric]=float(np.max(np.abs(values[metric]-saved[metric].to_numpy())))
            np.testing.assert_allclose(values[metric],saved[metric],rtol=2e-4,atol=2e-5)
        checks.append({'job_id':job,'sigma':sigma,'image_ids':selected_ids,'max_absolute_difference':diffs})
        return result
    for sigma,number in [(.3,9),(.5,10)]:
        code=f'{round(sigma*10):02d}'
        rows=[clean[:,0],noisy[sigma][:,0],gaussian_denoise(noisy[sigma],filters['width_by_sigma'][str(sigma)])[:,0]]
        labels=['原图','含噪图','高斯滤波']
        for prefix,label,suffix in [('fc_baseline','FC z16','z16'),('cae_baseline','CAE','c16'),
                                    ('cae_residual','CAE+R','c16'),('cae_gradient','CAE+G','c16'),
                                    ('cae_residual_gradient','CAE+R+G','c16')]:
            job=f'{prefix}_s{code}_{suffix}_seed42'
            rows.append(infer(job,noisy[sigma],clean,ids,sigma)[:,0]);labels.append(label)
        np.savez_compressed(DATA/f'recon_sigma{code}.npz',images=np.stack(rows),image_ids=ids,labels=labels)
        image_grid(rows,labels,ids,list(range(10)),f'fig{number:02d}_reconstruction_s{code}',f'σ={sigma}时的固定样本重建对比',
                   f'所有模型使用相同输入，训练和测试σ={sigma}；展示seed42的第20轮模型。每类样本按固定规则选择，非按重建效果挑选。R为残差预测，G为梯度约束。定量结论基于三种子全测试集。')
    fail=read(REPORT/'failure_selection.json'); ids=[x['image_id'] for x in fail['samples']]
    clean=torch.stack([dataset[i]['clean'] for i in ids]);x=fixed_evaluation_noise(clean,torch.tensor(ids),.5,seed=19300)
    base=infer('cae_baseline_s05_c16_seed42',x,clean,ids,.5)
    combo=infer(fail['job_id'],x,clean,ids,.5)
    rows=[clean[:,0],x[:,0],base[:,0],combo[:,0],(combo-clean).abs()[:,0]]
    labels=['原图','含噪图','CAE','CAE+R+G','绝对误差']
    np.savez_compressed(DATA/'failure_images.npz',images=np.stack(rows),image_ids=ids,labels=labels)
    image_grid(rows,labels,ids,[x['label'] for x in fail['samples']],'fig11_failures','组合模型的低SSIM失败案例',
               '固定采用组合模型seed42在σ=0.5测试中SSIM最低的8例，同分按ID排序。误差图为该组合模型相对原图的绝对像素误差，统一色标[0,1]；不代表随机样本或所有方法的总体错误分布。',error=True)
    (DATA/'sample_inference_check.json').write_text(json.dumps({'scope':'116 image-model forwards only; CPU visualization, not training or full evaluation',
         'torch':torch.__version__,'reference_environment':'Kaggle torch 2.11.0+cu128',
         'tolerance':{'rtol':2e-4,'atol':2e-5},'checks':checks},ensure_ascii=False,indent=2),encoding='utf-8')

def tables():
    table('tab01_settings','数据与训练设置','4.1–4.3',['项目','设置'],[
        ['数据集','MNIST，28×28灰度图，像素归一化至[0,1]'],
        ['划分','训练55,000；验证5,000；测试10,000'],['划分种子','2026；所有训练种子共享固定划分'],
        ['训练随机种子','42、2026、3407'],['优化器','Adam；学习率0.001；批量128'],
        ['训练预算','20轮；以第20轮last.pt为主要检查点'],['固定噪声','σ=0.3或0.5；训练时动态生成并截断'],
        ['混合噪声','逐样本σ∈[0.1,0.5)均匀采样'],['验证噪声','固定；噪声种子9300'],
        ['测试噪声','σ=0.0至0.6，步长0.1；种子19300'],['梯度权重','仅验证集选参；最终λ=0.1'],
        ['评价','MSE、逐图平均PSNR、SSIM、梯度MAE'],['计算环境','Tesla T4；PyTorch 2.11.0+cu128'],
        ['不确定性','3个训练种子的均值±样本标准差']],
        'SSIM：data_range=1，win_size=7，均匀窗口；评估重建统一截断到[0,1]。训练种子波动不等于对新噪声实现的不确定性。',[.23,.77])
    configs=[]
    for path in sorted((WORK/'runs').glob('*/config.json')):
        config=read(path)
        if config['purpose']!='formal' or config['seed']!=42: continue
        model=config['model']; train=config['training']; variant=config['experiment']['variant']
        configs.append([variant,'FC' if model['type']=='fc' else ('条件CAE' if model['type']=='cae_conditioned' else 'CAE'),
                        str(model.get('latent_dim','c16')),'混合' if train['noise_mode']=='mixed' else str(train['sigma']),
                        '是' if model.get('residual',False) else '否',config['loss']['gradient_weight'],3])
    assert len(configs)==19
    # Full IDs are retained in editable appendix CSV; shorter display labels fit paper width.
    matrix=pd.DataFrame(configs,columns=['配置ID','结构','z或通道','训练σ','残差','λ','种子数'])
    matrix.to_csv(DATA/'experiment_matrix_full.csv',index=False,encoding='utf-8-sig')
    display_rows=[[f'E{i:02d}',*r[1:]] for i,r in enumerate(configs,1)]
    table('tab02_matrix','正式实验配置矩阵','4.2',['编号','结构','z / c','训练σ','残差','λ','种子数'],display_rows,
          '19种配置×3种子=57次正式训练；另有3次梯度权重选参。c16为基础通道数，FC列为瓶颈维度z。完整编号与配置ID映射保存在source_data/experiment_matrix_full.csv。')
    pd.DataFrame([{'编号':f'E{i:02d}',**dict(zip(matrix.columns,r))} for i,r in enumerate(configs,1)]).to_csv(DATA/'experiment_matrix_full.csv',index=False,encoding='utf-8-sig')
    table('tab03_selection','梯度损失权重的验证集选择','4.2.3',['λ','验证MSE ↓','验证SSIM ↑','采用'],[
        [f"{c['weight']:.2f}",f"{c['validation_mse']:.6f}",f"{c['validation_ssim']:.6f}",'是' if c['weight']==selection['weight'] else '否'] for c in selection['candidates']],
        '仅直接CAE、σ=0.3、seed42的第20轮验证结果；先最大化SSIM，同分依次比较较低MSE和较小λ。三行不是多种子统计，未使用测试集。')
    rows=[]
    for sigma in [.3,.5]:
        code=f'{round(sigma*10):02d}'
        for v,label in [('noisy_input','含噪输入'),('gaussian_filter','高斯滤波'),(f'fc_baseline_s{code}_z16','FC z16'),
                        (f'fc_baseline_s{code}_z64','FC z64'),(f'cae_baseline_s{code}_c16','CAE'),(f'cae_gradient_s{code}_c16','CAE+G')]:
            r=row(v,sigma);rows.append([sigma,label,fmt(r,'MSE',5),fmt(r,'PSNR',3),fmt(r,'SSIM',4)])
    table('tab04_main','匹配噪声下的主要测试结果','5.1.2',['σ','方法','MSE ↓','PSNR / dB ↑','SSIM ↑'],rows,
          '学习模型按三种子报告均值±样本标准差；训练σ与测试σ匹配。两种确定性参照仅评一次，因此不报告种子标准差。PSNR先逐图计算再平均。',[.06,.15,.29,.25,.25])
    rows=[]
    for sigma in [.3,.5]:
        for z in [8,16,32,64]:
            r=row(f'fc_baseline_s{round(sigma*10):02d}_z{z}',sigma)
            rows.append([sigma,z,int(r.parameters),fmt(r,'PSNR',3),fmt(r,'SSIM',4)])
    table('tab05_bottleneck','全连接瓶颈维度对照','5.2',['σ','z','参数量','PSNR / dB ↑','SSIM ↑'],rows,
          '训练与测试噪声匹配；三种子均值±样本标准差。增加z时参数量亦变化，不能将收益解释成与容量无关的因素。')
    rows=[]
    for sigma in [.1,.3,.5,.6]:
        for v,label in [('cae_baseline_s03_c16','固定0.3'),('cae_baseline_s05_c16','固定0.5'),('cae_mixed_c16','混合'),('cae_conditioned_mixed_c16','混合+已知σ')]:
            r=row(v,sigma);rows.append([sigma,label,fmt(r,'PSNR',3),fmt(r,'SSIM',4)])
    table('tab06_robustness','噪声适应策略的测试对照','5.3',['测试σ','训练策略','PSNR / dB ↑','SSIM ↑'],rows,
          '三种子均值±样本标准差。正文列代表噪声档，完整0.0–0.6结果保留在附录数据。已知σ来自实验设定，不含噪声估计过程。')
    rows=[]
    for sigma in [.3,.5]:
        for prefix,label in [('baseline','A 基础'),('residual','B +R'),('gradient','C +G'),('residual_gradient','D +R+G')]:
            r=row(f'cae_{prefix}_s{round(sigma*10):02d}_c16',sigma)
            rows.append([sigma,label,fmt(r,'MSE',5),fmt(r,'PSNR',3),fmt(r,'SSIM',4)])
    table('tab07_ablation','残差预测与梯度约束的四格消融','5.4',['σ','变体','MSE ↓','PSNR / dB ↑','SSIM ↑'],rows,
          '三种子均值±样本标准差。R为残差预测，G为λ=0.1梯度约束。原始配对差值、梯度MAE和因素交互效应保留在附录数据。',[.06,.15,.29,.25,.25])
    rows=[]
    for variant,label in [(f'fc_baseline_s03_z{z}',f'FC z{z}') for z in [8,16,32,64]]+[
            ('cae_baseline_s03_c16','CAE'),('cae_gradient_s03_c16','CAE+G'),('cae_residual_s03_c16','CAE+R'),
            ('cae_residual_gradient_s03_c16','CAE+R+G'),('cae_conditioned_mixed_c16','条件CAE')]:
        part=by_seed.loc[(by_seed.variant==variant)&np.isclose(by_seed.sigma,.3)]
        times=part.network_ms_per_image
        rows.append([label,int(part.parameters.iloc[0]),f'{times.mean():.4f} ± {times.std(ddof=1):.4f}',fmt(row(variant,.3),'PSNR',3)])
    table('tab08_cost','参数规模与网络推理耗时','6.1',['方法','参数量','摊销耗时 / (ms/图)','σ=0.3 PSNR / dB'],rows,
          'Tesla T4，batch128，预热10次、重复50次并同步GPU；仅网络前向的批处理摊销耗时，不含加噪、数据读写和指标计算，不等于单图延迟。条件CAE使用混合训练，其余为固定σ=.3；三种子均值±样本标准差。')

def finish():
    registry.sort(key=lambda r:r['id'])
    cards=[]
    for item in registry:
        links=f'<a href="{item["pdf"]}">矢量 PDF</a>'
        if item['kind']=='table': links+=f' · <a href="tables/{item["id"]}.csv">可编辑 CSV</a>'
        cards.append(f'<article id="{item["id"]}"><h2>{html.escape(item["id"]+"　"+item["title"])}</h2>'
                     f'<p class="section">建议位置：{html.escape(item["section"])} · {links}</p>'
                     f'<img src="{item["png"]}" alt="{html.escape(item["title"])}">'
                     f'<p>{html.escape(item["caption"])}</p></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MNIST论文图表素材</title>
<style>body{margin:36px auto;max-width:1000px;padding:0 24px;background:#f3f4f6;color:#17212c;font:16px/1.7 "Microsoft YaHei",sans-serif}h1{font-size:28px}h2{font-size:21px}article{background:white;padding:26px;margin:24px 0;border:1px solid #ddd}img{display:block;width:100%;height:auto}.section{color:#526070;font-size:14px}a{color:#235789}p{margin:12px 0}</style>
<h1>MNIST图像降噪论文图表素材</h1><p>最终测试已冻结。正文图表以真实记录为依据；误差棒和阴影表示三个训练种子的样本标准差。表格提供可编辑CSV，最终Word将排成原生三线表。此页用于浏览素材，图号将在正文定稿时统一编号。</p>'''+''.join(cards)+'</html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    text='# 论文图表清单\n\n来源：最终测试包；完整性验收见outputs/review/final_test。原始结果未覆盖。\n\n'
    for r in registry:
        text+=f"## {r['id']} {r['title']}\n\n章节：{r['section']}。\n\n{r['caption']}\n\n"
    (OUT/'README.md').write_text(text,encoding='utf-8')
    manifest={'source_archive_sha256':audit['archive_sha256'], 'font':FONT,
              'training_performed':False, 'full_evaluation_performed':False,
              'sample_forward_checks':'source_data/sample_inference_check.json',
              'figures':sum(r['kind']=='figure' for r in registry),'tables':sum(r['kind']=='table' for r in registry),
              'assets':registry,'files_sha256':{str(p.relative_to(OUT)).replace('\\','/'):sha(p)
                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PAPER ASSETS READY',OUT,flush=True)

if __name__=='__main__':
    diagrams()
    curves()
    numerical_figures()
    examples()
    tables()
    finish()
