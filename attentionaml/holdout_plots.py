"""Publication figures from saved held-out predictions; no fitting or tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.text import Text

plt.rcParams.update({"font.weight":"bold", "axes.labelweight":"bold", "axes.titleweight":"bold", "figure.titleweight":"bold", "font.size":14, "axes.titlesize":20, "axes.labelsize":16, "xtick.labelsize":13, "ytick.labelsize":13, "legend.fontsize":13, "figure.titlesize":20})

SUBTYPE_ORDER = ['KMT2Ar', 'RUNX1::RUNX1T1', 'CBFB::MYH11', 'NPM1', 'NUP98r', 'CEBPA', 'UBTF', 'GLISr', 'DEK::NUP214', 'GATA1', 'KMT2A::PTD', 'ETS family', 'BCL11B', 'MECOM', 'HOXr', 'RBM15::MKL1', 'PICALM::MLLT10', 'KAT6Ar', 'MNX1', 'CBFB-GDXY', 'RUNX1::RUNX1T1-like']
MODEL_ORDER = ["RanAML", "MLP", "AttentionAML"]

def ordered_models(models):
    return [m for m in MODEL_ORDER if m in models] + [m for m in models if m not in MODEL_ORDER]

ALIASES = {"KMT2A-PTD": "KMT2A::PTD"}

def canonical(label):
    return ALIASES.get(label, label)

def ordered_classes(classes):
    normalized = [canonical(c) for c in classes]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Class aliases collapse distinct model outputs; resolve the class vocabulary first.")
    return [c for c in SUBTYPE_ORDER if c in normalized] + [c for c in normalized if c not in SUBTYPE_ORDER]

def bold_text(fig):
    fig.canvas.draw()
    for text in fig.findobj(match=Text):
        text.set_fontweight("bold")


def confusion_values(true, predicted, classes):
    counts=confusion_matrix(true,predicted,labels=classes)
    support=counts.sum(axis=1)
    proportions=np.divide(counts,support[:,None],out=np.zeros_like(counts,dtype=float),where=support[:,None]!=0)
    return counts,proportions,support


def draw_confusion(ax, counts, proportions, support, classes, title):
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#eeeeee')
    values=np.ma.array(proportions,mask=np.broadcast_to(support[:,None]==0,counts.shape))
    im=ax.imshow(values,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
    n=len(classes)
    ax.set_xticks(range(n),classes,rotation=60,ha='right',fontsize=13)
    ax.set_yticks(range(n),[f'{c} (n={s})' for c,s in zip(classes,support)],fontsize=13)
    ax.set_xlabel('Predicted molecular subtype');ax.set_ylabel('True molecular subtype')
    ax.set_title(title,pad=12)
    for i,j in zip(*np.nonzero(counts)):
        ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=12,
                color='white' if proportions[i,j]>=0.55 else '#222222')
    return im


def save_figure(fig,base):
    bold_text(fig)
    fig.savefig(str(base)+'.png',dpi=300,bbox_inches='tight')
    fig.savefig(str(base)+'.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_results(output,classes,model_names):
    classes=ordered_classes(classes)
    model_names=ordered_models(model_names)
    output=Path(output);directory=output/'figures';directory.mkdir(exist_ok=True)
    matrices=[];predictions=[]
    for name in model_names:
        df=pd.read_csv(output/name/'test_predictions.csv')
        df['true_label']=df.true_label.map(canonical)
        df['predicted_label']=df.predicted_label.map(canonical)
        unknown=(set(df.true_label)|set(df.predicted_label))-set(classes)
        if unknown: raise ValueError(f'Prediction labels absent from class vocabulary: {unknown}')
        predictions.append(df)
        counts,proportions,support=confusion_values(df.true_label,df.predicted_label,classes)
        matrices.append((counts,proportions,support))
        pd.DataFrame(counts,index=classes,columns=classes).rename_axis('true_subtype').to_csv(output/name/'confusion_counts.csv')
        pd.DataFrame(np.where(support[:,None]>0,proportions,np.nan),index=classes,columns=classes).rename_axis('true_subtype').to_csv(output/name/'confusion_row_proportions.csv')
        size=max(8,len(classes)*0.60)
        fig,ax=plt.subplots(figsize=(size+2,size),layout='constrained')
        im=draw_confusion(ax,counts,proportions,support,classes,f'{name}: held-out test set (n={len(df)})')
        fig.colorbar(im,ax=ax,label='Proportion within true subtype',shrink=0.75)
        save_figure(fig,directory/f'{name}_confusion_matrix')
    size=max(8,len(classes)*0.55)
    fig,axes=plt.subplots(1,len(model_names),figsize=((size+2)*len(model_names),size+2),squeeze=False,layout='constrained')
    for ax,name,(counts,proportions,support) in zip(axes[0],model_names,matrices):
        im=draw_confusion(ax,counts,proportions,support,classes,name)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Proportion within true subtype',shrink=0.7)
    fig.suptitle('Held-out test set | annotations: nonzero sample counts; colors: row proportions')
    save_figure(fig,directory/'confusion_matrices_all_models')
    # Shared bin edges and probability axes; counts remain interpretable for rare errors.
    for field,caption,filename in [
        ('predicted_probability','Probability assigned to the predicted subtype','prediction_probability_distribution'),
        ('true_class_probability','Probability assigned to the true subtype','true_class_probability_distribution')]:
        fig,axes=plt.subplots(1,len(model_names),figsize=(6.6*len(model_names),5.5),squeeze=False,sharex=True,sharey=True,layout='constrained')
        for ax,name,df in zip(axes[0],model_names,predictions):
            correct=df.true_label.eq(df.predicted_label)
            for mask,label,color in [(correct,'Correct','#2878B5'),(~correct,'Incorrect','#D95319')]:
                values=df.loc[mask,field].to_numpy()
                ax.hist(values,bins=np.linspace(0,1,21),histtype='step',linewidth=2,color=color,label=f'{label} (n={len(values)})')
            ax.set(title=name,xlabel=caption.replace(' to the ', ' to the\n'),xlim=(0,1));ax.legend(frameon=False)
            ax.spines[['top','right']].set_visible(False)
        axes[0,0].set_ylabel('Number of test samples')
        fig.suptitle('Held-out test prediction probabilities')
        save_figure(fig,directory/filename)

    # Subtype grouping is by the true label, including misclassified samples.
    summary=[]
    for field,caption,filename in [
        ('predicted_probability','Probability assigned to predicted subtype','predicted_probability_by_subtype'),
        ('true_class_probability','Probability assigned to true subtype','true_class_probability_by_subtype')]:
        height=max(6,len(classes)*0.46+2)
        fig,axes=plt.subplots(1,len(model_names),figsize=(6.6*len(model_names)+3,height),
                              squeeze=False,sharex=True,sharey=True,layout='constrained')
        for ax,name,df in zip(axes[0],model_names,predictions):
            jitter=np.random.default_rng(0)
            for i,subtype in enumerate(classes):
                subset=df.loc[df.true_label==subtype]
                values=subset[field].to_numpy(dtype=float)
                if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
                    raise ValueError('Invalid probability in saved predictions.')
                correct=subset.true_label.eq(subset.predicted_label).to_numpy()
                if len(values):
                    ax.boxplot([values],positions=[i],vert=False,widths=0.55,showfliers=False,
                        patch_artist=True,manage_ticks=False,
                        boxprops={'facecolor':'#eeeeee','edgecolor':'#555555'},
                        medianprops={'color':'black','linewidth':1.6},
                        whiskerprops={'color':'#555555'},capprops={'color':'#555555'})
                    offsets=jitter.uniform(-0.18,0.18,len(values))
                    for mask,label,color in [(correct,'Correct','#2878B5'),(~correct,'Incorrect','#D95319')]:
                        ax.scatter(values[mask],i+offsets[mask],s=17,color=color,alpha=0.65,zorder=3)
                else:
                    ax.text(0.5,i,'No test samples',ha='center',va='center',fontsize=13,color='#666666')
                summary.append(dict(model=name,subtype=subtype,probability_type=field,n=len(values),
                    n_correct=int(correct.sum()),n_incorrect=int((~correct).sum()),
                    median=float(np.median(values)) if len(values) else np.nan,
                    q25=float(np.quantile(values,.25)) if len(values) else np.nan,
                    q75=float(np.quantile(values,.75)) if len(values) else np.nan))
            for label,color in [('Correct','#2878B5'),('Incorrect','#D95319')]:
                ax.scatter([],[],s=22,color=color,label=label)
            ax.set_title(name,pad=48);ax.set_xlabel(caption.replace(' to ', ' to\n'),fontsize=16);ax.set_xlim(-.025,1.025)
            ax.set_ylim(len(classes)-.5,-.5)
            ax.set_yticks(range(len(classes)),classes,fontsize=13)
            ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
            ax.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False,fontsize=13)
        axes[0,0].set_ylabel('True molecular subtype')
        fig.suptitle('Held-out test probabilities by true subtype | boxes: median and IQR; points: individual samples')
        save_figure(fig,directory/filename)
    pd.DataFrame(summary).to_csv(output/'subtype_probability_summary.csv',index=False)
