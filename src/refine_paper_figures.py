"""Scientific diagrams and additional analyses of frozen MNIST results.

No model training, inference, hyperparameter selection, or data alteration.
Run with the existing analysis environment. Exports vector PDF/SVG and 400 dpi PNG.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper_assets'
FIG=OUT/'figures'; DATA=OUT/'source_data'
INK='#252b33'; BLUE='#31688e'; ORANGE='#c27c35'; GOLD='#b29642'
PALETTE=[BLUE,ORANGE,'#638347','#985b7a','#555555']
for f in ('C:/Windows/Fonts/simsun.ttc','C:/Windows/Fonts/times.ttf'):
    if Path(f).exists(): font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family':['Times New Roman','SimSun','DejaVu Sans'],
 'font.size':10,'axes.labelsize':10,'axes.titlesize':10.5,'legend.fontsize':9,
 'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
manifest=json.loads((OUT/'manifest.json').read_text('utf-8'))
updates=[]

def write(fig,key,title,caption,sources,section):
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{key}.{ext}',dpi=400,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    a={'id':key,'kind':'figure','title':title,'caption':caption,'section':section,
       'png':f'figures/{key}.png','pdf':f'figures/{key}.pdf','svg':f'figures/{key}.svg',
       'sources':sources}
    updates.append(a);print('Created',key,flush=True)

def arr(ax,x1,y1,x2,y2,color=INK,style='-',lw=.9):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
      mutation_scale=8,linewidth=lw,color=color,linestyle=style,shrinkA=0,shrinkB=0))

def rect(ax,x,y,w,h,text='',color=BLUE,fill='white',size=10):
    ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor=fill,edgecolor=color,lw=.9))
    if text: ax.text(x,y,text,ha='center',va='center',fontsize=size,color=INK)

def architecture():
    fig,axes=plt.subplots(2,1,figsize=(8.8,5.8),gridspec_kw={'height_ratios':[1,1.15]})
    ax=axes[0];ax.set(xlim=(-.6,6.6),ylim=(-.7,1.4));ax.axis('off')
    ax.text(-.5,1.21,'(a) 全连接降噪自编码器',fontsize=11,fontweight='bold')
    dims=['784','256','64',r'$z$','64','256','784']; heights=[1,.82,.64,.42,.64,.82,1]
    for i,(d,h) in enumerate(zip(dims,heights)):
        col=BLUE if i<3 else (GOLD if i==3 else ORANGE)
        ys=np.linspace(.32-h/2,.32+h/2,5 if i!=3 else 3)
        for yy in ys:ax.add_patch(Circle((i,yy),.037,edgecolor=col,facecolor='white',lw=.9))
        if i<6:
            ys2=np.linspace(.32-heights[i+1]/2,.32+heights[i+1]/2,5 if i+1!=3 else 3)
            for j,yy in enumerate(ys):
                for k,zz in enumerate(ys2):
                    if j==k or j in (0,len(ys)-1):ax.plot([i+.04,i+.96],[yy,zz],lw=.35,color='#a5adb4',alpha=.72,zorder=-1)
        ax.text(i,-.34,d,ha='center',va='center',fontsize=10)
    ax.text(1,.94,'Encoder',ha='center',color=BLUE)
    ax.text(3,.94,'Latent code',ha='center',color=GOLD)
    ax.text(5,.94,'Decoder',ha='center',color=ORANGE)
    ax.text(3,-.62,r'$z\in\{8,16,32,64\}$    Hidden: ReLU    Output: Sigmoid',ha='center',fontsize=9)
    ax=axes[1];ax.set(xlim=(-.7,6.9),ylim=(-.85,1.5));ax.axis('off')
    ax.text(-.6,1.28,'(b) 卷积降噪自编码器',fontsize=11,fontweight='bold')
    shapes=['2×28×28','16×14×14','32×7×7','64×7×7','32×14×14','16×28×28','1×28×28']
    hs=[.9,.65,.46,.46,.65,.9,.9];ops=['Conv 3×3 / 2','Conv 3×3 / 2','Conv 3×3 / 1','TConv 4×4 / 2','TConv 4×4 / 2','Conv 3×3 / 1']
    for i,(shape,h) in enumerate(zip(shapes,hs)):
        col=BLUE if i<=3 else ORANGE
        layers=2 if i in (0,6) else 4
        for j in reversed(range(layers)):
            dx=.035*j;dy=.027*j
            ax.add_patch(Rectangle((i-.24+dx,.26-h/2+dy),.43,h,facecolor='white' if j==0 else '#f1f3f5',edgecolor=col,lw=.8))
        ax.text(i+.015,-.36,shape,ha='center',fontsize=9)
        if i<6:
            arr(ax,i+.4,.28,i+.73,.28)
            ax.text(i+.5,.91,ops[i],ha='center',va='center',fontsize=7.8,rotation=0)
    ax.text(3,-.64,'Image + auxiliary map → spatial compression → feature decoding',ha='center',fontsize=9)
    ax.text(3,-.82,'Conv / TConv labels: kernel / stride; padding = 1; ReLU after first five layers.',ha='center',fontsize=8.3)
    fig.subplots_adjust(hspace=.28)
    write(fig,'fig02_architecture','全连接与卷积自编码器结构',
      '圆点为全连接层的示意节点，不对应实际节点数量；图中数字给出真实维度。卷积特征图按C×H×W标注，叠层数量仅作示意。辅助通道为0或已知σ图；末层线性，评价输出截断至[0,1]。', ['src/models.py'],'3.2')

def computation():
    fig,axes=plt.subplots(2,1,figsize=(8.8,5.6),gridspec_kw={'height_ratios':[1.05,1]})
    ax=axes[0];ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    ax.text(.05,3.67,'(a) 相同骨干下的两种重建路径',fontsize=11,fontweight='bold')
    for yy,col,net in [(2.8,BLUE,r'$f_\theta$: Encoder → Decoder'),(1.35,ORANGE,r'$r_\theta$: Encoder → Decoder')]:
        rect(ax,.65,yy,1.1,.65,r'$[y,a]$',color=col)
        rect(ax,3.25,yy,2.8,.65,net,color=col,size=10)
        arr(ax,1.2,yy,1.85,yy,color=col)
    arr(ax,4.65,2.8,7.25,2.8,color=BLUE)
    rect(ax,8.6,2.8,2.7,.65,r'Direct: $\hat{x}=f_\theta(y,a)$',color=BLUE,size=10)
    arr(ax,4.65,1.35,6.12,1.35,color=ORANGE)
    ax.add_patch(Circle((6.4,1.35),.28,fill=False,ec=ORANGE,lw=.9))
    ax.text(6.4,1.35,'−',ha='center',va='center',fontsize=16,color=ORANGE)
    ax.plot([.65,.65,6.4],[1.02,.52,.52],color=ORANGE,lw=.85)
    arr(ax,6.4,.52,6.4,1.07,color=ORANGE)
    ax.text(4.6,.61,r'input $y$',fontsize=9,color=ORANGE)
    arr(ax,6.68,1.35,7.25,1.35,color=ORANGE)
    rect(ax,8.6,1.35,2.7,.65,r'Residual: $\hat{x}=y-r_\theta(y,a)$',color=ORANGE,size=10)
    ax.text(.05,.12,r'$a=0$: ordinary model; $a=\sigma\mathbf{1}$: known-noise conditioning. Weights are trained separately.',fontsize=9)
    ax=axes[1];ax.set(xlim=(0,10),ylim=(0,3.6));ax.axis('off')
    ax.text(.05,3.32,'(b) 像素与梯度双路径监督',fontsize=11,fontweight='bold')
    rect(ax,.7,1.65,1.3,.85,r'$\hat{x},x$',color=INK)
    ax.plot([1.35,1.8,1.8],[1.65,1.65,2.5],color=INK,lw=.85);arr(ax,1.8,2.5,2.5,2.5)
    ax.plot([1.8,1.8],[1.65,.9],color=INK,lw=.85);arr(ax,1.8,.9,2.5,.9)
    rect(ax,3.7,2.5,2.4,.7,r'Pixel MSE',color=BLUE)
    rect(ax,3.7,.9,2.4,.9,'Horizontal / vertical\nfinite differences',color=ORANGE,size=9)
    arr(ax,4.9,.9,5.5,.9,color=ORANGE);rect(ax,6.35,.9,1.7,.7,r'$L_1$ distance',color=ORANGE)
    ax.plot([4.9,8.05,8.05],[2.5,2.5,1.93],color=BLUE,lw=.85)
    ax.plot([7.2,8.05,8.05],[.9,.9,1.37],color=ORANGE,lw=.85)
    ax.text(7.58,1.02,r'$\times\lambda$',color=ORANGE,fontsize=10)
    ax.add_patch(Circle((8.05,1.65),.28,fill=False,ec=INK,lw=.9));ax.text(8.05,1.65,'+',ha='center',va='center',fontsize=14)
    arr(ax,8.33,1.65,9,1.65);ax.text(9.15,1.65,r'$\mathcal{L}$',ha='left',va='center',fontsize=15)
    ax.text(.05,.1,'A: MSE   B: residual + MSE   C: MSE + gradient   D: residual + MSE + gradient',fontsize=9)
    fig.subplots_adjust(hspace=.28)
    write(fig,'fig03_variants','重建与损失的计算路径',
      '上图两种输出路径分别训练，残差相减顺序为y−rθ。下图在未截断预测上计算损失；λ=0或0.1。辅助通道条件化属于独立实验族，不代表已执行所有交叉组合。',['src/models.py','src/losses.py'],'3.3')

def protocol():
    fig,ax=plt.subplots(figsize=(8.8,5.2));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,6.4))
    lanes=[(1.5,'训练 Training',BLUE),(5,'验证 Validation',ORANGE),(8.5,'测试 Test',INK)]
    for x,t,c in lanes:
        ax.text(x,5.72,t,ha='center',color=c,fontsize=11,fontweight='bold')
        ax.plot([x-1.4,x+1.4],[5.5,5.5],color=c,lw=1.1)
    rect(ax,1.5,4.95,2.65,.65,'55,000 images',BLUE)
    rect(ax,5,4.95,2.65,.65,'5,000 images',ORANGE)
    rect(ax,8.5,4.95,2.65,.65,'10,000 images',INK)
    ax.text(3.25,6.12,'Official training set 60,000 → fixed split (seed 2026)',ha='center',fontsize=9)
    rect(ax,1.5,3.75,2.65,.95,'3 pilot runs\nCAE, σ=0.3, seed 42',BLUE,size=9.5)
    rect(ax,5,3.75,2.65,.95,'Fixed noise (seed 9300)\nSelect λ by validation SSIM',ORANGE,size=9.5)
    arr(ax,1.5,4.62,1.5,4.23,color=BLUE);arr(ax,5,4.62,5,4.23,color=ORANGE)
    arr(ax,2.83,3.75,3.67,3.75,color=ORANGE)
    rect(ax,1.5,2.05,2.65,1.1,'19 configs × 3 seeds\n20 epochs; batch 128\nAdam; lr = 0.001',BLUE,size=9.5)
    arr(ax,1.5,3.27,1.5,2.61,color=BLUE)
    ax.plot([5,5,2.85],[3.27,2.72,2.72],color=ORANGE,lw=.9);arr(ax,2.85,2.72,2.35,2.6,color=ORANGE)
    ax.text(3.8,2.83,'Freeze λ=0.1',color=ORANGE,ha='center',fontsize=9)
    rect(ax,8.5,2.05,2.65,1.1,'Fixed test noise (seed 19300)\nσ=0.0, 0.1, …, 0.6\n57 models + 2 references',INK,size=9.5)
    arr(ax,8.5,4.62,8.5,2.61,color=INK)
    arr(ax,2.83,2.05,7.17,2.05,color=BLUE)
    ax.text(5,2.19,'Freeze epoch-20 weights (last.pt)',ha='center',fontsize=9,color=BLUE)
    rect(ax,8.5,.6,2.65,.8,'413 evaluation groups\nMetrics · reconstruction · ablation',INK,size=9)
    arr(ax,8.5,1.49,8.5,1.01)
    ax.text(.15,.65,'No test feedback to training or model selection.\nAll models share the same split and evaluation noise.',fontsize=9,color=INK)
    write(fig,'fig01_protocol','训练验证测试分离的实验协议',
       '训练与验证来自官方训练集的固定划分，测试来自官方测试集。3次预实验仅用于验证选权重；正式19配置×3种子使用第20轮权重，测试结果不回流调参。',['configs/experiment_plan.yaml','src/experiments.py'],'4.1')

def cost_plot():
    df=pd.read_csv(DATA/'results_by_seed.csv');df=df[np.isclose(df.sigma,.3)&df.seed.notna()]
    specs=[('fc_baseline_s03_z8','FC 8'),('fc_baseline_s03_z16','FC 16'),('fc_baseline_s03_z32','FC 32'),('fc_baseline_s03_z64','FC 64'),('cae_baseline_s03_c16','CAE'),('cae_gradient_s03_c16','CAE+G'),('cae_residual_s03_c16','CAE+R'),('cae_residual_gradient_s03_c16','CAE+R+G')]
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.8));records=[]
    for i,(v,label) in enumerate(specs):
        a=df[df.variant==v];assert len(a)==3,v
        col=PALETTE[0] if i<4 else PALETTE[(i-3)%5]
        y=a.PSNR.mean();ys=a.PSNR.std(ddof=1)
        x=a.parameters.iloc[0]/1000;z=a.network_ms_per_image.mean()
        axes[0].errorbar(x,y,yerr=ys,fmt='o' if i<4 else 's',color=col,capsize=2,ms=5)
        axes[1].errorbar(z,y,xerr=a.network_ms_per_image.std(ddof=1),yerr=ys,fmt='o' if i<4 else 's',color=col,capsize=2,ms=5,label=label)
        offsets=[(-13,-12),(7,-11),(7,-3),(7,5),(8,-12),(8,10),(8,3),(8,-13)]
        for ax,xx in zip(axes,[x,z]):ax.annotate(label,(xx,y),xytext=offsets[i],textcoords='offset points',fontsize=8)
        records.append(dict(variant=v,parameters=a.parameters.iloc[0],PSNR_mean=y,PSNR_sd=ys,network_ms_mean=z,network_ms_sd=a.network_ms_per_image.std(ddof=1)))
    for ax,t in zip(axes,['(a) 参数量与重建质量','(b) 批量前向代价与重建质量']):
        ax.set_title(t,loc='left');ax.set_ylabel('PSNR / dB');ax.grid(alpha=.17,lw=.5);ax.set_ylim(15.4,25.1)
    axes[0].set(xlabel='Parameters / 10³',xlim=(0,525));axes[1].set(xlabel='Network time / (ms per image)',xlim=(.0084,.0158))
    axes[1].ticklabel_format(axis='x',style='plain',useOffset=False);fig.tight_layout(w_pad=2)
    pd.DataFrame(records).to_csv(DATA/'quality_cost.csv',index=False)
    write(fig,'fig13_quality_cost','重建质量与计算代价的关系',
      '训练/测试σ=0.3，点为3种子均值，误差棒为样本标准差；参数量为实际计数。T4上batch=128，预热10次、重复50次并同步计时，仅统计网络前向的每图摊销耗时。',['source_data/results_by_seed.csv'],'6.1')

def distributions(work):
    specs=[('fc_baseline','FC z16','z16'),('cae_baseline','CAE','c16'),('cae_gradient','CAE+G','c16'),('cae_residual','CAE+R','c16'),('cae_residual_gradient','CAE+R+G','c16')]
    data={};rows=[]
    for sigma in (.3,.5):
        for prefix,label,suffix in specs:
            frames=[]
            for seed in (42,2026,3407):
                job=f'{prefix}_s{round(sigma*10):02d}_{suffix}_seed{seed}'
                f=pd.read_csv(work/'evaluation/test'/job/'per_image.csv.gz')
                a=f[np.isclose(f.sigma,sigma)].sort_values('image_id')
                assert len(a)==10000 and a.image_id.is_unique
                assert np.array_equal(a.image_id.to_numpy(),np.arange(10000))
                frames.append(a[['PSNR','SSIM']].to_numpy())
            data[(sigma,label)]=np.stack(frames).mean(0)
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.6))
    for ax,sigma in zip(axes,(.3,.5)):
        for i,(_,label,_) in enumerate(specs):
            a=np.sort(data[(sigma,label)][:,1]);y=np.arange(1,10001)/10000
            ax.plot(a,y,color=PALETTE[i],lw=1.2,label=label,ls=['-', '-', '--','-.',':'][i])
        ax.set(title=f'σ={sigma}',xlabel='Per-image SSIM (mean over 3 seeds)',ylabel='Empirical cumulative fraction',xlim=(0,1),ylim=(0,1));ax.grid(alpha=.17,lw=.5)
    axes[0].legend(frameon=False,loc='upper left',fontsize=8.5);fig.tight_layout()
    write(fig,'fig14_ssim_distribution','逐图结构相似度的经验累积分布',
      '每幅测试图像先对3个训练种子的SSIM取平均，再对10,000幅图构建经验CDF。曲线在同一阈值处越低，低于该阈值的图像比例越少。该图描述固定测试集，不把3种子视为30,000幅独立图像。',['evaluation/test/*/per_image.csv.gz'],'5.5')
    fig,axes=plt.subplots(2,2,figsize=(8.4,6))
    for r,sigma in enumerate((.3,.5)):
        delta=data[(sigma,'CAE+G')]-data[(sigma,'CAE')]
        for c,metric in enumerate(('PSNR','SSIM')):
            v=delta[:,c];ax=axes[r,c]
            # Fixed bins cover the full observed range; no observations are removed.
            ax.hist(v,bins=70,color=BLUE,alpha=.83,edgecolor='white',lw=.25)
            ax.axvline(0,color=INK,lw=.9,ls='--');ax.axvline(v.mean(),color=ORANGE,lw=1.1)
            ax.set_title(f'σ={sigma}: CAE+G − CAE',loc='left')
            ax.set(xlabel=f'Paired Δ{metric}'+(' / dB' if c==0 else ''),ylabel='Number of test images')
            ax.text(.97,.95,f'Δ > 0: {(v>0).mean()*100:.1f}%\nMean: {v.mean():+.4f}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
            rows.append(dict(sigma=sigma,metric=metric,n_images=10000,fraction_positive=float((v>0).mean()),mean_delta=float(v.mean()),median_delta=float(np.median(v))))
    fig.tight_layout(h_pad=2)
    pd.DataFrame(rows).to_csv(DATA/'per_image_gradient_deltas.csv',index=False)
    write(fig,'fig15_paired_distribution','梯度约束的逐图配对收益分布',
      '以相同图像、相同测试噪声和相同训练种子比较CAE+G与CAE，再对3个种子的差值取平均。虚线为零，实线为全测试集均值；直方图包含全部10,000幅图。正差比例不等同于显著性检验或独立数据集泛化率。',['evaluation/test/cae_baseline*/per_image.csv.gz','evaluation/test/cae_gradient*/per_image.csv.gz'],'5.4')
    np.savez_compressed(DATA/'per_image_seed_averages.npz',**{f's{int(s*10):02d}_{l.replace(" ","_").replace("+","_")}':a for (s,l),a in data.items()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work-dir',type=Path,default=ROOT/'dist/formal_results_20261022T004235964922Z/formal')
    a=parser.parse_args();architecture();computation();protocol();cost_plot();distributions(a.work_dir)
    keep=[v for v in manifest['assets'] if v['id'] not in {u['id'] for u in updates}]
    manifest['assets']=keep+updates;manifest['figures']=sum(v['kind']=='figure' for v in manifest['assets'])
    manifest['files_sha256']={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Refined 3 diagrams; added 3 evidence figures; no training or inference.')
if __name__=='__main__':main()
