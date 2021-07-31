import torch
import torch.optim
import numpy as np
import argparse
import torch.nn as nn
import model
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)

def trimap_transform(trimap):
    h, w = trimap.shape[0], trimap.shape[1]
    clicks = np.zeros((h, w, 6))
    for k in range(2):
        if (np.count_nonzero(trimap[:, :, k]) > 0):
            dt_mask = -dt(1 - trimap[:, :, k]) ** 2
            L = 320
            clicks[:, :, 3 * k] = np.exp(dt_mask / (2 * ((0.02 * L) ** 2)))
            clicks[:, :, 3 * k + 1] = np.exp(dt_mask / (2 * ((0.08 * L) ** 2)))
            clicks[:, :, 3 * k + 2] = np.exp(dt_mask / (2 * ((0.16 * L) ** 2)))
    return clicks

def genwmap(qsss,www):
    h,w=qsss+www,qsss+www
    www1=np.ones([h,w],np.float32)*www
    ws=np.zeros([h,w],np.float32)
    for xx in range(h):
        for yy in range(w):
            ws[xx,yy]=min(xx,yy,h-xx-1,w-yy-1)+1
    ws=ws/(www1+1)
    return ws

if __name__ == '__main__':
    IMG_SCALE = 1. / 255
    IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    segmodel = model.LFPNet()
    segmodel.load_state_dict(torch.load('model.pth',map_location='cpu'),strict=False)
    segmodel=segmodel.cuda()
    segmodel.eval()
    for idx,x in enumerate(os.listdir('./merged/')):
        img=cv2.imread('./merged/'+x)
        raw_h,raw_w,_=img.shape
        alphaall=np.zeros((raw_h,raw_w),np.float32)
        img = cv2.imread('./merged/' + x)
        h1_, w1_, _ = img.shape
        psss=512
        qsss=1024-512
        wsss=qsss+psss
        wsss2=wsss//2
        wx=(w1_-1-psss) //qsss+1
        hx=(h1_-1-psss) //qsss+1
        alphalist=[]
        newh=hx*qsss+psss
        neww=wx*qsss+psss
        raw_h, raw_w, _ = img.shape
        imgzys=img.copy()
        alls=[]
        alltri=[]
        ph1=(newh-raw_h)//2
        ph2=newh-raw_h-ph1
        pw1=(neww-raw_w)//2
        pw2=neww-raw_w-pw1
        allp=set(list(np.arange(0,16)))
        tp = np.zeros((newh, neww,16), np.float32)
        wp = np.zeros((newh, neww, 16), np.float32)
        apPP = np.zeros((newh, neww, 16), np.float32)
        allp = set(list(np.arange(0, 16)))
        wsm = genwmap(qsss, psss)
        for x11 in [0]:
            for y11 in [0]:
                pwh=[0,0,0,0]
                pwh=[ph1,ph2,pw1,pw2]
                imgbp = cv2.copyMakeBorder(imgzys, pwh[0], pwh[1], pwh[2], pwh[3], cv2.BORDER_REPLICATE)
                tri = cv2.imread('./trimap/' + x, cv2.IMREAD_GRAYSCALE)
                trizys=tri.copy()
                tribp = cv2.copyMakeBorder(tri, pwh[0], pwh[1], pwh[2], pwh[3], cv2.BORDER_CONSTANT)
                imgbpp=cv2.copyMakeBorder(imgbp, wsss2,wsss2,wsss2,wsss2, cv2.BORDER_REPLICATE)
                tribpp=cv2.copyMakeBorder(tribp, wsss2,wsss2,wsss2,wsss2, cv2.BORDER_CONSTANT)
                alls = []
                for px in range(hx):
                    for py in range(wx):
                        nnns=0
                        for zzz in range(16):
                            if np.sum(tp[px*qsss:px*qsss+wsss,py*qsss:py*qsss+wsss,zzz])==0:
                                nnns=zzz
                                break
                        wp[px*qsss: px*qsss+wsss, py*qsss:py*qsss+wsss,nnns]=wsm
                        tp[px*qsss:px*qsss+wsss,py*qsss:py*qsss+wsss,nnns] = 1
                        alls = []
                        for fl1, fl2 in zip([0], [0]):
                            if fl1 >= 0:
                                imgf = imgbp[px * qsss:px * qsss + wsss, py * qsss:py * qsss + wsss].copy()
                                trif = tribp[px * qsss:px * qsss + wsss, py * qsss:py * qsss + wsss].copy()
                                imgrr = cv2.flip(imgf, fl1)
                                trirr = cv2.flip(trif, fl1)
                                imgss2=cv2.flip(imgzys.copy(), fl1)
                                triss2=cv2.flip(trizys.copy(), fl1)
                                img4x2 = imgbpp[px*qsss:(px)*qsss+wsss+wsss,py*qsss:(py)*qsss+wsss+wsss].copy()
                                tri4x2 = tribpp[px*qsss:(px)*qsss+wsss+wsss,py*qsss:(py)*qsss+wsss+wsss].copy()
                                img4x2 = cv2.flip(img4x2, fl1)
                                tri4x2 = cv2.flip(tri4x2, fl1)
                            else:
                                imgf = imgbp[px * qsss:px * qsss + wsss, py * qsss:py * qsss + wsss].copy()
                                trif = tribp[px * qsss:px * qsss + wsss, py * qsss:py * qsss + wsss].copy()
                                imgrr = imgf.copy()
                                trirr = trif.copy()
                                imgss2 = imgzys.copy()
                                triss2 = trizys.copy()
                                img4x2 = imgbpp[px*qsss:(px)*qsss+wsss+wsss,py*qsss:(py)*qsss+wsss+wsss].copy()
                                tri4x2 = tribpp[px*qsss:(px)*qsss+wsss+wsss,py*qsss:(py)*qsss+wsss+wsss].copy()
                            for fr1, fr2 in zip([0], [2]):
                                if fr1 >= 0:
                                    imgrr3 = cv2.rotate(imgrr.copy(), fr1)
                                    trirr3 = cv2.rotate(trirr.copy(), fr1)
                                    imgss3 = cv2.rotate(imgss2.copy(), fr1)
                                    triss3 = cv2.rotate(triss2.copy(), fr1)
                                    img4x3 = cv2.rotate(img4x2.copy(), fr1)
                                    tri4x3 = cv2.rotate(tri4x2.copy(), fr1)
                                else:
                                    imgrr3= imgrr.copy()
                                    trirr3= trirr.copy()
                                    imgss3= imgss2.copy()
                                    triss3= triss2.copy()
                                    img4x3=img4x2.copy()
                                    tri4x3=tri4x2.copy()
                                img=imgrr3
                                tri=trirr3
                                imgss=imgss3
                                triss=triss3
                                img4x=img4x3
                                tri4x=tri4x3
                                tritemp = np.zeros([*tri.shape, 2], np.float32)
                                tritemp[:, :, 0] = (tri == 0)
                                tritemp[:, :, 1] = (tri == 255)
                                sixc = trimap_transform(tritemp)
                                sixc = np.transpose(sixc, [2, 0, 1])
                                tritemp = np.zeros([*tri4x.shape, 2], np.float32)
                                tritemp[:, :, 0] = (tri4x == 0)
                                tritemp[:, :, 1] = (tri4x == 255)
                                sixc4x = trimap_transform(tritemp)
                                sixc4x = np.transpose(sixc4x, [2, 0, 1])
                                h_, w_ = tri.shape
                                tri2 = np.array(tri, np.float32) / 255.
                                tri2 = tri2[np.newaxis, np.newaxis, :, :]
                                tri24x = np.array(tri4x, np.float32) / 255.
                                tri24x = tri2[np.newaxis, np.newaxis, :, :]
                                mattinginput = ((img[:, :, ::-1] / 255.) - IMG_MEAN) / IMG_STD
                                mattinginput4x = ((img4x[:, :, ::-1] / 255.) - IMG_MEAN) / IMG_STD
                                raw = img[:, :, ::-1] / 255.
                                raw = np.transpose(raw, [2, 0, 1])[None, :, :, :]
                                h_, w_ = tri.shape
                                trixs = np.zeros((1, 3, h_, w_), np.float32)
                                trixs[0, 0] = (tri == 0)
                                trixs[0, 1] = (tri == 128)
                                trixs[0, 2] = (tri == 255)
                                ntrixs = trixs
                                trixs = torch.from_numpy(trixs)
                                h_, w_ = tri4x.shape
                                tris4x = np.zeros((1, 3, h_, w_), np.float32)
                                tris4x[0, 0] = (tri4x == 0)
                                tris4x[0, 1] = (tri4x == 128)
                                tris4x[0, 2] = (tri4x == 255)
                                tris4x = torch.from_numpy(tris4x)
                                mattinginput = np.transpose(mattinginput, (2, 0, 1)).astype(np.float32)
                                mattinginput = np.array(mattinginput)[np.newaxis, :, :, :]
                                mattinginput4x = np.transpose(mattinginput4x, (2, 0, 1)).astype(np.float32)
                                mattinginput4x = np.array(mattinginput4x)[np.newaxis, :, :, :]
                                i1 = torch.from_numpy(mattinginput)
                                i2 = torch.from_numpy(mattinginput4x)
                                sixc = sixc[None, :, :, :]
                                sixc4x = sixc4x[None, :, :, :]
                                sixc2 = torch.from_numpy(sixc)
                                sixc24x = torch.from_numpy(sixc4x)
                                i2i = torch.cat([i2, tris4x, sixc24x], 1).float().cuda()
                                i1i = torch.cat([i1, trixs, sixc2], 1).float().cuda()
                                rawi = torch.from_numpy(raw).float().cuda()
                                with torch.no_grad():
                                    preda= segmodel(i1i, i2i, rawi)
                                preda = preda.detach().cpu()
                                ap = preda[0, 0].numpy() * ntrixs[0, 1] + ntrixs[0, 2]
                                a1 = ap
                                if fr2 >= 0:
                                    a1 = cv2.rotate(a1, fr2)
                                if fl2 >= 0:
                                    a1 = cv2.flip(a1, fl2)
                                alls.append(a1)
                        a1 = np.array(sum(alls) * 255. / len(alls))
                        alls=[]
                        apPP[px * qsss:px * qsss + wsss, py * qsss:py * qsss + wsss,nnns] = a1
        palpha=np.sum(apPP*wp,2)/ np.sum(wp,2)
        palpha=np.clip(palpha,0,255)
        palpha=np.array(palpha,np.uint8)
        wholealpha=palpha[ph1:ph1+raw_h,pw1:pw1+raw_w]
        wholealpha[trizys==0]=0
        wholealpha[trizys==255]=255
        cv2.imwrite('./alpha/' + x, wholealpha)

