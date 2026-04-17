import streamlit as st
import numpy as np
import cv2
import tifffile
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import json
import csv
import time
import datetime
import pandas as pd

# ── Page config ──
st.set_page_config(
    page_title='Bacterial Detection using AI | EP7 Group | University of Bradford',
    page_icon='https://www.researchgate.net/profile/Flora-Romeo/publication/233833232/figure/fig4/AS:300038552080388@1448546173601/S-aureus-strain-detected-by-fluorescence-microscopy.png',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Google Fonts + CSS ──
st.markdown('''
<link href='https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;0,700;1,300&family=DM+Mono:wght@400;500&display=swap' rel='stylesheet'>
<style>
:root {
    --navy:       #060d1a;
    --navy-mid:   #0b1a2e;
    --navy-card:  #0e2038;
    --teal:       #00c9b1;
    --teal-dim:   #00877a;
    --gold:       #c9a84c;
    --cold-white: #ddeaf7;
    --muted:      #6a8faa;
    --border:     rgba(0,201,177,0.15);
    --font-d: 'Playfair Display', Georgia, serif;
    --font-b: 'Source Sans 3', sans-serif;
    --font-m: 'DM Mono', monospace;
}
html, body, [class*='css'] { font-family: var(--font-b) !important; background-color: var(--navy) !important; color: var(--cold-white) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 1380px !important; }

/* Hero */
.hero {
    position: relative; width: 100%; min-height: 380px;
    background:
        linear-gradient(180deg, rgba(6,13,26,0.15) 0%, rgba(6,13,26,0.70) 45%, var(--navy) 100%),
''', unsafe_allow_html=True)

# ── Model definition ──
def build_unet(input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    c1 = layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    c1 = layers.Conv2D(16,3,activation='relu',padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1); p1 = layers.Dropout(0.1)(p1)
    c2 = layers.Conv2D(32,3,activation='relu',padding='same')(p1)
    c2 = layers.Conv2D(32,3,activation='relu',padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2); p2 = layers.Dropout(0.1)(p2)
    c3 = layers.Conv2D(64,3,activation='relu',padding='same')(p2)
    c3 = layers.Conv2D(64,3,activation='relu',padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3); p3 = layers.Dropout(0.2)(p3)
    c4 = layers.Conv2D(128,3,activation='relu',padding='same')(p3)
    c4 = layers.Conv2D(128,3,activation='relu',padding='same')(c4)
    p4 = layers.MaxPooling2D()(c4); p4 = layers.Dropout(0.2)(p4)
    c5 = layers.Conv2D(256,3,activation='relu',padding='same')(p4)
    c5 = layers.Conv2D(256,3,activation='relu',padding='same')(c5)
    u6 = layers.Conv2DTranspose(128,2,strides=2,padding='same')(c5)
    u6 = layers.concatenate([u6,c4])
    c6 = layers.Conv2D(128,3,activation='relu',padding='same')(u6)
    c6 = layers.Conv2D(128,3,activation='relu',padding='same')(c6); c6 = layers.Dropout(0.2)(c6)
    u7 = layers.Conv2DTranspose(64,2,strides=2,padding='same')(c6)
    u7 = layers.concatenate([u7,c3])
    c7 = layers.Conv2D(64,3,activation='relu',padding='same')(u7)
    c7 = layers.Conv2D(64,3,activation='relu',padding='same')(c7); c7 = layers.Dropout(0.2)(c7)
    u8 = layers.Conv2DTranspose(32,2,strides=2,padding='same')(c7)
    u8 = layers.concatenate([u8,c2])
    c8 = layers.Conv2D(32,3,activation='relu',padding='same')(u8)
    c8 = layers.Conv2D(32,3,activation='relu',padding='same')(c8); c8 = layers.Dropout(0.1)(c8)
    u9 = layers.Conv2DTranspose(16,2,strides=2,padding='same')(c8)
    u9 = layers.concatenate([u9,c1])
    c9 = layers.Conv2D(16,3,activation='relu',padding='same')(u9)
    c9 = layers.Conv2D(16,3,activation='relu',padding='same')(c9); c9 = layers.Dropout(0.1)(c9)
    outputs = layers.Conv2D(1,1,activation='sigmoid')(c9)
    return models.Model(inputs,outputs)

@st.cache_resource
def load_model():
    m = build_unet()
    m.load_weights('/content/drive/MyDrive/bacteria_model/best_model.h5')
    return m

# ── Helpers ──
def preprocess_image(img):
    if len(img.shape)==3: img=img[:,:,0]
    img=cv2.resize(img.astype(np.float32),(256,256))
    img=(img-img.min())/(img.max()-img.min()+1e-8)
    return img

def count_watershed(pred_binary,min_dist=5):
    try:
        distance=distance_transform_edt(pred_binary)
        coords=peak_local_max(distance,min_distance=min_dist,labels=pred_binary.astype(bool))
        local_max=np.zeros(distance.shape,dtype=bool)
        local_max[tuple(coords.T)]=True
        markers,_=ndimage.label(local_max)
        labels=watershed(-distance,markers,mask=pred_binary.astype(bool))
        return int(labels.max())
    except Exception:
        _,count=ndimage.label(pred_binary); return int(count)

def make_overlay(img,pred_binary):
    rgb=np.stack([img,img,img],axis=-1)
    rgb=(rgb*255).astype(np.uint8)
    ov=rgb.copy(); ov[pred_binary==1]=[0,201,177]
    return cv2.addWeighted(rgb,0.60,ov,0.40,0)

def fig_to_bytes(fig):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',facecolor='#060d1a',dpi=150)
    buf.seek(0); return buf.getvalue()

def make_fig(img,cmap,caption):
    fig,ax=plt.subplots(figsize=(4,4))
    ax.imshow(img,cmap=cmap); ax.axis('off')
    ax.set_title(caption,color='#6a8faa',fontsize=8,pad=7,fontfamily='monospace')
    fig.patch.set_facecolor('#060d1a'); fig.tight_layout(pad=0.3)
    return fig

def gen_csv(data):
    buf=io.StringIO()
    if not data: return ''
    w=csv.DictWriter(buf,fieldnames=data[0].keys())
    w.writeheader(); w.writerows(data)
    return buf.getvalue()

def gen_report(img_proc,pred_bin,pred,wcount,bcount,cov,conf,fname,thr):
    fig=plt.figure(figsize=(18,9),facecolor='#060d1a')
    fig.suptitle('BacteriaAI  —  Analysis Report  |  EP7 Group  |  University of Bradford',
                 fontsize=14,color='white',y=0.97,fontweight='bold',fontfamily='serif')
    axs=[fig.add_subplot(2,4,i+1) for i in range(8)]
    axs[0].imshow(img_proc,cmap='gray'); axs[0].axis('off'); axs[0].set_title('Original',color='#6a8faa',fontsize=9)
    axs[1].imshow(pred_bin,cmap='hot'); axs[1].axis('off'); axs[1].set_title('Predicted Mask',color='#6a8faa',fontsize=9)
    axs[2].imshow(make_overlay(img_proc,pred_bin)); axs[2].axis('off'); axs[2].set_title('Overlay',color='#6a8faa',fontsize=9)
    im=axs[3].imshow(pred,cmap='RdYlGn',vmin=0,vmax=1); axs[3].axis('off')
    axs[3].set_title('Confidence',color='#6a8faa',fontsize=9); plt.colorbar(im,ax=axs[3],fraction=0.046)
    mn=['Dice','IoU','Precision','Recall']; mv=[0.8109,0.6820,0.7358,0.9063]
    cl=['#00c9b1','#00a896','#007d70','#005a51']
    bars=axs[4].bar(mn,mv,color=cl,edgecolor='#060d1a')
    axs[4].set_ylim(0,1.15); axs[4].set_facecolor('#0e2038')
    axs[4].tick_params(colors='#6a8faa',labelsize=8); axs[4].set_title('Model Metrics',color='#6a8faa',fontsize=9)
    for b,v in zip(bars,mv): axs[4].text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.0%}',ha='center',color='white',fontsize=8)
    axs[5].bar(['Watershed','Basic'],[wcount,bcount],color=['#00c9b1','#1a3055'],edgecolor='#060d1a')
    axs[5].set_facecolor('#0e2038'); axs[5].tick_params(colors='#6a8faa')
    axs[5].set_title('Bacteria Count',color='#6a8faa',fontsize=9)
    for i,v in enumerate([wcount,bcount]): axs[5].text(i,v+0.2,str(v),ha='center',color='white',fontsize=13,fontweight='bold')
    axs[6].pie([cov,100-cov],labels=['Bacteria','Background'],colors=['#00c9b1','#1a3055'],
               autopct='%1.1f%%',textprops={'color':'white','fontsize':8},
               wedgeprops={'edgecolor':'#060d1a','linewidth':1.5})
    axs[6].set_title('Coverage',color='#6a8faa',fontsize=9)
    axs[7].axis('off')
    ts=datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')
    txt=('SUMMARY\n'+'-'*24+'\n'
         +f'File:       {fname[:22]}\n'
         +f'Date:       {ts}\n'
         +'-'*24+'\n'
         +f'Bacteria:   {wcount} cells\n'
         +f'Coverage:   {cov:.1f}%\n'
         +f'Confidence: {conf:.1f}%\n'
         +f'Threshold:  {thr}\n'
         +'-'*24+'\n'
         +'Model:      U-Net\n'
         +'Species:    S. aureus\n'
         +'Dice:       81.1%\n'
         +'-'*24+'\n'
         +'Group:      EP7\n'
         +'Team:       Kiran, Bandika, Jenish\n'
         +'Bradford University')
    axs[7].text(0.05,0.95,txt,transform=axs[7].transAxes,fontsize=8,va='top',
                fontfamily='monospace',color='#c8dff0',
                bbox=dict(boxstyle='round,pad=0.8',facecolor='#0e2038',edgecolor='#00c9b1',linewidth=1.2))
    plt.tight_layout(rect=[0,0,1,0.95]); return fig

# ── Session state ──
if 'history' not in st.session_state: st.session_state.history=[]

# ════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════
st.markdown('''
<div class='hero'>
  <div class='hero-label'>COS6032-E &nbsp;/&nbsp; Industrial AI Project &nbsp;/&nbsp; University of Bradford &nbsp;/&nbsp; Final Year Project &nbsp;2025&ndash;2026</div>
  <h1 class='hero-title'>Bacterial &mdash;<br><em>Detection &amp; Quantification System Using AI</em></h1>
  <p class='hero-desc'>
    An end-to-end deep learning pipeline for automated segmentation and counting of
    <em>Staphylococcus aureus</em> in brightfield and fluorescence microscopy images.
    Developed by EP7 Group as a final year project at the University of Bradford,
    addressing the global challenge of rapid, accurate bacterial diagnostics.
  </p>
  <div class='hero-tags'>
    <span class='tag gold'>EP7 Group &nbsp;&mdash;&nbsp; Kiran &middot; Bandika &middot; Jenish</span>
  </div>
</div>
''', unsafe_allow_html=True)

# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.58rem;letter-spacing:0.22em;text-transform:uppercase;color:#00c9b1;padding:14px 0 10px">EP7 Group Project</div>', unsafe_allow_html=True)
    st.image('https://www.researchgate.net/publication/233833232/figure/fig4/AS:300038552080388@1448546173601/S-aureus-strain-detected-by-fluorescence-microscopy.png',
             caption='Staphylococcus aureus strain detected by fluorescence microscopy.',
             use_column_width=True)
    st.markdown('''
    <div class='s-block'>
      <div class='s-lbl'>Academic Supervisor:Dr Kulvinder Panesar</div>
      <div class='s-name'></div>
      <div class='s-lbl' style='margin-top:10px;'>Client:Dr Maria Katsikogianni</div>
      <div class='s-name'></div>
      <div class='s-name'></div>
      <div class='s-lbl' style='margin-top:10px;'></div>
    </div>
    <div class='s-block'>
      <div class='s-lbl'>Project Team &mdash; EP7 Group:</div>
      <div class='s-name'>Kiran Nyaupane</div>
      <div class='s-name'>Bandika Dhital</div>
      <div class='s-name'>Jenish Dani</div>
      <div class='s-name'></div>
      <div class='s-lbl' style='margin-top:10px;'></div>
      <div class='s-lbl' style='margin-top:10px;'></div>
    </div>
    <div class='s-block'>
      <div class='s-lbl'>Module:</div>
      <div style='font-size:0.82rem;color:#c8dff0;line-height:1.7;'>
        COS6032-E<br>Industrial AI Project<br>University of Bradford<br>2025 &ndash; 2026
      </div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="s-lbl">Detection Settings</div>', unsafe_allow_html=True)
    threshold   = st.slider('Detection Threshold', 0.10, 0.90, 0.50, 0.05,
                            help='Confidence cutoff for classifying a pixel as bacteria. Higher = stricter.')
    min_dist    = st.slider('Min Cell Separation (px)', 3, 15, 5,
                            help='Minimum pixel distance between individual cell centres.')
    show_overlay = st.toggle('Show colour overlay', value=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('''
    <div class='s-lbl'>Model Performance:</div>
    <div class='s-row'><span class='s-key'>Dice Coefficient</span>        -<span class='s-v'>81.1%</span></div>
    <div class='s-row'><span class='s-key'>IoU Score</span>               -<span class='s-v'>68.2%</span></div>
    <div class='s-row'><span class='s-key'>Precision</span>               -<span class='s-v'>73.6%</span></div>
    <div class='s-row'><span class='s-key'>Recall</span>                  -<span class='s-v'>90.6%</span></div>
    <div class='s-row'><span class='s-key'>Count Error Reduction</span>   -<span class='s-v'>63%</span></div>
    <div class='s-row'><span class='s-key'>Parameters</span>              -<span class='s-v'>1.94M</span></div>
    <div class='s-row'><span class='s-key'>Training Epochs</span>         -<span class='s-v'>100</span></div>
    ''', unsafe_allow_html=True)

# ════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════
tab1,tab2,tab3,tab4 = st.tabs(['Analyse Image','Model Performance','Session History','About the Project'])

# ── TAB 1: ANALYSE ──
with tab1:
    st.markdown('<div class="sec-title">Upload Microscopy Image</div><hr class="sec-rule">', unsafe_allow_html=True)
    st.markdown('Accepted formats: <code>.tif</code> &nbsp; <code>.tiff</code> &nbsp; <code>.png</code> &nbsp; <code>.jpg</code> &mdash; brightfield or fluorescence', unsafe_allow_html=True)
    uploaded = st.file_uploader('', type=['tif','tiff','png','jpg','jpeg'], label_visibility='collapsed')
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(('.tif','.tiff')):
                img_array=tifffile.imread(uploaded)
            else:
                img_array=np.array(Image.open(uploaded).convert('L'))
        except Exception as e:
            st.error(f'Could not load image: {e}'); st.stop()
        img_proc=preprocess_image(img_array)
        with st.spinner('Running inference...'):
            mdl=load_model()
            inp=img_proc[np.newaxis,...,np.newaxis]
            pred=mdl.predict(inp,verbose=0)[0,:,:,0]
            pred_bin=(pred>threshold).astype(np.float32)
            _,basic=ndimage.label(pred_bin)
            wcount=count_watershed(pred_bin,min_dist)
            cov=float(np.sum(pred_bin))/(256*256)*100
            conf=float(np.mean(pred[pred>threshold]))*100 if np.any(pred>threshold) else 0.0
            ts=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f'<div class="r-banner">Analysis complete &nbsp;&mdash;&nbsp; {uploaded.name} &nbsp;&mdash;&nbsp; {ts}</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Detection Results</div><hr class="sec-rule">', unsafe_allow_html=True)
        st.markdown(f'''
        <div class='m-grid'>
          <div class='m-card'><div class='m-num'>{wcount}</div><div class='m-lbl'>Bacteria Detected</div><div class='m-sub'>Watershed algorithm</div></div>
          <div class='m-card'><div class='m-num'>{cov:.1f}%</div><div class='m-lbl'>Image Coverage</div><div class='m-sub'>Bacteria pixel area</div></div>
          <div class='m-card'><div class='m-num'>{conf:.0f}%</div><div class='m-lbl'>Avg Confidence</div><div class='m-sub'>Detected regions only</div></div>
          <div class='m-card'><div class='m-num'>{basic}</div><div class='m-lbl'>Basic Count</div><div class='m-sub'>Connected components</div></div>
        </div>''', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Visualisation</div><hr class="sec-rule">', unsafe_allow_html=True)
        v1,v2,v3=st.columns(3,gap='medium')
        with v1:
            st.markdown('<div class="panel-lbl">Original Microscopy Image</div>', unsafe_allow_html=True)
            fig=make_fig(img_proc,'gray','Input  |  greyscale  |  256 x 256 px')
            st.image(Image.open(io.BytesIO(fig_to_bytes(fig))),use_column_width=True); plt.close(fig)
        with v2:
            st.markdown('<div class="panel-lbl">AI Predicted Segmentation Mask</div>', unsafe_allow_html=True)
            fig=make_fig(pred_bin,'hot',f'Binary mask  |  threshold = {threshold}')
            st.image(Image.open(io.BytesIO(fig_to_bytes(fig))),use_column_width=True); plt.close(fig)
        with v3:
            if show_overlay:
                st.markdown('<div class="panel-lbl">Overlay  (teal = bacteria detected)</div>', unsafe_allow_html=True)
                st.image(make_overlay(img_proc,pred_bin),use_column_width=True)
            else:
                st.markdown('<div class="panel-lbl">Confidence Heatmap</div>', unsafe_allow_html=True)
                fig,ax=plt.subplots(figsize=(4,4))
                im=ax.imshow(pred,cmap='RdYlGn',vmin=0,vmax=1); ax.axis('off')
                plt.colorbar(im,ax=ax,fraction=0.046); fig.patch.set_facecolor('#060d1a')
                st.image(Image.open(io.BytesIO(fig_to_bytes(fig))),use_column_width=True); plt.close(fig)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Export Results</div><hr class="sec-rule">', unsafe_allow_html=True)
        result={
            'filename':uploaded.name,'timestamp':ts,
            'bacteria_count_watershed':wcount,'bacteria_count_basic':basic,
            'coverage_percent':round(cov,2),'avg_confidence_percent':round(conf,2),
            'threshold':threshold,'min_distance_px':min_dist,
            'model':'U-Net','species':'Staphylococcus aureus',
            'dice_score':0.8109,'iou_score':0.6820,'precision':0.7358,'recall':0.9063,
            'group':'EP7','team':'Kiran, Bandika, Jenish',
            'supervisor':'Dr Kulvinder Panesar','client':'Dr Maria Katsikogianni',
            'institution':'University of Bradford','module':'COS6032-E'
        }
        e1,e2,e3,e4=st.columns(4,gap='small')
        with e1:
            st.download_button('Download JSON',json.dumps(result,indent=2),
                f'bacteria_{uploaded.name}.json','application/json',use_container_width=True)
        with e2:
            st.download_button('Download CSV',gen_csv([result]),
                f'bacteria_{uploaded.name}.csv','text/csv',use_container_width=True)
        with e3:
            rep=gen_report(img_proc,pred_bin,pred,wcount,basic,cov,conf,uploaded.name,threshold)
            st.download_button('Download Report PNG',fig_to_bytes(rep),
                f'report_{uploaded.name}.png','image/png',use_container_width=True)
            plt.close(rep)
        with e4:
            mb=io.BytesIO()
            Image.fromarray((pred_bin*255).astype(np.uint8)).save(mb,format='PNG')
            st.download_button('Download Mask PNG',mb.getvalue(),
                f'mask_{uploaded.name}.png','image/png',use_container_width=True)
        st.session_state.history.append(result)
        st.info(f'Result saved to session history  ({len(st.session_state.history)} total)')
    else:
        st.markdown('''
        <div style='border:1px dashed rgba(0,201,177,0.2);background:rgba(14,32,56,0.4);
                    padding:44px 36px;text-align:center;'>
          <div style='font-family:DM Mono,monospace;font-size:0.72rem;letter-spacing:0.14em;
                      text-transform:uppercase;color:#00c9b1;margin-bottom:14px;'>
            Drop a microscopy image here or click to browse
          </div>
          <div style='color:#6a8faa;font-size:0.88rem;max-width:520px;margin:0 auto;line-height:1.8;'>
            The AI model will automatically segment every bacterium, produce a binary mask,
            count individual cells using the Watershed algorithm, and report confidence metrics.
            Supports brightfield and fluorescence microscopy images.
          </div>
        </div>''', unsafe_allow_html=True)

# ── TAB 2: PERFORMANCE ──
with tab2:
    st.markdown('<div class="sec-title">Model Evaluation Results</div><hr class="sec-rule">', unsafe_allow_html=True)
    col_l,col_r=st.columns([1,1],gap='large')
    with col_l:
        st.markdown('<div class="sec-sub">Segmentation Metrics &mdash; Test Set (5 unseen images)</div>', unsafe_allow_html=True)
        for name,val in [('Dice Coefficient',0.8109),('IoU Score',0.6820),('Precision',0.7358),('Recall',0.9063)]:
            st.markdown(f'<div class="p-row"><div class="p-name">{name}</div><div class="p-track"><div class="p-fill" style="width:{val*100:.1f}%"></div></div><div class="p-val">{val*100:.1f}%</div></div>',unsafe_allow_html=True)
        st.markdown('<br><div class="sec-sub">Training Configuration</div>', unsafe_allow_html=True)
        for k,v in [('Architecture','U-Net  (encoder-decoder + skip connections)'),
                    ('Parameters','1,940,817'),('Epochs','100  |  early stopping: patience 20'),
                    ('Loss','Dice loss + Binary cross-entropy'),('Optimiser','Adam  (lr = 1e-4)'),
                    ('Batch size','8'),('Training set','67 images after augmentation'),
                    ('Best val Dice','0.8273  (epoch 100)'),('Training time','~15 min  |  Google Colab T4 GPU')]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(0,201,177,0.07);font-size:0.82rem;"><span style="color:#6a8faa;font-family:DM Mono,monospace;font-size:0.7rem;">{k}</span><span style="color:#c8dff0;">{v}</span></div>',unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="sec-sub">Bacteria Counting &mdash; Basic vs Watershed</div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(6,4))
        imgs=['Img 1','Img 2','Img 3','Img 4','Img 5']
        actual=[40,34,70,48,41]; basic2=[33,23,54,38,31]; wshd=[46,32,65,53,39]
        x=np.arange(5)
        ax.bar(x-0.25,actual,0.25,label='Ground Truth',color='#00c9b1',alpha=0.9)
        ax.bar(x,      basic2,0.25,label='Basic Count', color='#1a3055',alpha=0.9)
        ax.bar(x+0.25, wshd,  0.25,label='Watershed',  color='#00877a',alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(imgs,color='#6a8faa',fontsize=9)
        ax.tick_params(colors='#6a8faa',labelsize=9)
        ax.set_ylabel('Cell Count',color='#6a8faa',fontsize=9)
        ax.spines[['top','right','left','bottom']].set_color('#1a3055')
        ax.legend(facecolor='#0e2038',labelcolor='#c8dff0',fontsize=8,edgecolor='#00c9b1',framealpha=0.8)
        ax.set_facecolor('#0e2038'); fig.patch.set_facecolor('#060d1a')
        st.pyplot(fig); plt.close(fig)
        st.markdown('''
        <div style='background:#0e2038;border-left:3px solid #00c9b1;
                    padding:14px 18px;margin-top:12px;font-size:0.82rem;line-height:2;color:#c8dff0;'>
          Basic counting total error &nbsp;&nbsp; <strong style='color:#e05c5c;'>54</strong><br>
          Watershed total error &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong style='color:#00c9b1;'>20</strong><br>
          Improvement &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong style='color:#00c9b1;'>63% reduction in counting error</strong>
        </div>''', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Validation Dice Coefficient &mdash; Training Progress (100 Epochs)</div>', unsafe_allow_html=True)
    epochs=[1,10,20,30,40,50,60,70,80,90,100]
    v_dice=[0.181,0.125,0.272,0.553,0.743,0.786,0.811,0.822,0.826,0.827,0.827]
    t_dice=[0.150,0.102,0.280,0.538,0.737,0.780,0.809,0.820,0.823,0.827,0.830]
    fig2,ax2=plt.subplots(figsize=(12,3.2))
    ax2.plot(epochs,t_dice,color='#1a3055',linewidth=2,label='Train Dice',linestyle='--')
    ax2.plot(epochs,v_dice,color='#00c9b1',linewidth=2.5,label='Validation Dice')
    ax2.fill_between(epochs,t_dice,v_dice,alpha=0.07,color='#00c9b1')
    ax2.set_xlim(1,100); ax2.set_ylim(0,1.0)
    ax2.set_xlabel('Epoch',color='#6a8faa',fontsize=9); ax2.set_ylabel('Dice Coefficient',color='#6a8faa',fontsize=9)
    ax2.tick_params(colors='#6a8faa',labelsize=9)
    ax2.spines[['top','right','left','bottom']].set_color('#1a3055')
    ax2.legend(facecolor='#0e2038',labelcolor='#c8dff0',fontsize=9,edgecolor='#00c9b1',framealpha=0.8)
    ax2.set_facecolor('#0e2038'); fig2.patch.set_facecolor('#060d1a')
    st.pyplot(fig2); plt.close(fig2)

# ── TAB 3: HISTORY ──
with tab3:
    st.markdown('<div class="sec-title">Session Analysis History</div><hr class="sec-rule">', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info('No analyses have been run this session. Upload an image in the Analyse tab.')
    else:
        st.markdown(f'<div class="sec-sub">{len(st.session_state.history)} image(s) analysed this session</div>', unsafe_allow_html=True)
        df=pd.DataFrame(st.session_state.history)
        display_cols=['filename','timestamp','bacteria_count_watershed','bacteria_count_basic','coverage_percent','avg_confidence_percent']
        st.dataframe(df[display_cols],use_container_width=True,hide_index=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        b1,b2,b3=st.columns(3,gap='small')
        with b1:
            st.download_button('Export All as JSON',json.dumps(st.session_state.history,indent=2),
                'all_results.json','application/json',use_container_width=True)
        with b2:
            st.download_button('Export All as CSV',gen_csv(st.session_state.history),
                'all_results.csv','text/csv',use_container_width=True)
        with b3:
            summary={'total_images':len(st.session_state.history),
                     'avg_bacteria':round(sum(r['bacteria_count_watershed'] for r in st.session_state.history)/len(st.session_state.history),1),
                     'avg_coverage':round(sum(r['coverage_percent'] for r in st.session_state.history)/len(st.session_state.history),2),
                     'session_date':datetime.datetime.now().strftime('%Y-%m-%d'),
                     'group':'EP7','team':'Kiran, Bandika, Jenish','model':'U-Net S. aureus'}
            st.download_button('Export Summary Stats',json.dumps(summary,indent=2),
                'session_summary.json','application/json',use_container_width=True)
        if st.button('Clear Session History'):
            st.session_state.history=[]
            st.rerun()

# ── TAB 4: ABOUT ──
with tab4:
    st.markdown('<div class="sec-title">About the Project</div><hr class="sec-rule">', unsafe_allow_html=True)
    st.markdown('''
    <div class='info-grid'>
      <div class='i-cell'><div class='i-lbl'>Module</div><div class='i-val'>COS6032-E &mdash; Industrial AI Project</div></div>
      <div class='i-cell'><div class='i-lbl'>Institution</div><div class='i-val'>University of Bradford</div></div>
      <div class='i-cell'><div class='i-lbl'>Academic Year</div><div class='i-val'>2025 &ndash; 2026</div></div>
      <div class='i-cell'><div class='i-lbl'>Group</div><div class='i-val'>EP7 &mdash; Kiran, Bandika, Jenish</div></div>
      <div class='i-cell'><div class='i-lbl'>Supervisor</div><div class='i-val'>Dr Kulvinder Panesar</div></div>
      <div class='i-cell'><div class='i-lbl'>Client</div><div class='i-val'>Dr Maria Katsikogianni</div></div>
    </div>
    ''', unsafe_allow_html=True)
    a1,a2=st.columns(2,gap='large')
    with a1:
        st.markdown('<div class="sec-sub" style="margin-top:28px;">The Clinical Problem</div>', unsafe_allow_html=True)
        st.markdown('''
        <div style='font-size:0.88rem;line-height:1.9;color:#c8dff0;'>
          <em>Staphylococcus aureus</em> is a Gram-positive bacterium responsible for infections
          ranging from minor skin conditions to life-threatening sepsis and pneumonia.
          It is listed as a priority pathogen by the World Health Organization due to its
          ability to form biofilms and acquire antibiotic resistance, including the
          methicillin-resistant strain MRSA.<br><br>
          Traditional diagnosis relies on manual microscopy, which is slow, labour-intensive,
          and highly skill-dependent. This problem is particularly severe in low- and
          middle-income countries where trained personnel are scarce, leading to diagnostic
          delays and overuse of antibiotics &mdash; accelerating antimicrobial resistance (AMR).<br><br>
          This project addresses that gap by developing an AI pipeline that automates
          bacterial detection, segmentation, and counting from microscopy images,
          making rapid quantification accessible without specialist expertise.
        </div>''', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub" style="margin-top:28px;">Dataset &mdash; DeepBacs</div>', unsafe_allow_html=True)
        st.markdown('''
        <div style='font-size:0.88rem;line-height:1.9;color:#c8dff0;'>
          The DeepBacs dataset (Ouyang et al., 2022) provides paired microscopy images
          and expert-annotated binary segmentation masks for <em>S. aureus</em>.
          Both brightfield and fluorescence modalities are included.
          The fluorescence images use Nile Red dye, which causes bacteria to emit bright
          signals under UV excitation, producing high-contrast images ideal for segmentation.
          All 28 training image-mask pairs passed quality checks with zero corrupt,
          missing, or empty files.
        </div>''', unsafe_allow_html=True)
    with a2:
        st.markdown('<div class="sec-sub" style="margin-top:28px;">Technical Approach</div>', unsafe_allow_html=True)
        for comp,tech in [
            ('Architecture','U-Net  (encoder-decoder with skip connections)'),
            ('Segmentation','Binary segmentation + Watershed separation'),
            ('Counting','Euclidean distance transform + Watershed'),
            ('Dataset','DeepBacs  &mdash;  Staphylococcus aureus'),
            ('Image types','Brightfield and fluorescence microscopy'),
            ('Augmentation','Horizontal + vertical flip  (28 &rarr; 84 samples)'),
            ('Training','Google Colab T4 GPU  |  TensorFlow 2.19'),
            ('Dashboard','Streamlit'),
            ('Framework','CDIO  &mdash;  Conceive, Design, Implement, Operate'),
        ]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid rgba(0,201,177,0.07);font-size:0.82rem;"><span style="color:#6a8faa;font-family:DM Mono,monospace;font-size:0.7rem;">{comp}</span><span style="color:#c8dff0;">{tech}</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="sec-sub" style="margin-top:28px;">Key Results</div>', unsafe_allow_html=True)
        st.markdown('''
        <div style='font-size:0.88rem;line-height:2.0;color:#c8dff0;'>
          Dice coefficient on unseen test images &nbsp;&nbsp; <strong style='color:#00c9b1;'>81.1%</strong><br>
          Recall (bacteria detected out of total) &nbsp;&nbsp; <strong style='color:#00c9b1;'>90.6%</strong><br>
          Watershed counting error reduction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong style='color:#00c9b1;'>63% vs basic method</strong><br>
          Best validation Dice (epoch 100) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong style='color:#00c9b1;'>82.7%</strong><br>
          Model parameters &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong style='color:#00c9b1;'>1,940,817</strong>
        </div>'''
        , unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        st.image('https://wwwn.cdc.gov/phil///PHIL_Images/7488/7488_lores.jpg',
                 caption='S. aureus biofilm formation  |  Scanning Electron Microscope  |  CDC / Janice Haney Carr',
                 use_column_width=True)