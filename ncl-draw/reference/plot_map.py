import Ngl, Nio
import os

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

f = Nio.open_file("/home/xingj/Documents/dlrsm/pm25/iRSM/data/GRIDCRO2D_cn27_2017001.nc","r")
lat2d = f.variables["LAT"][0,0,:,:]
lon2d = f.variables["LON"][0,0,:,:]
ncol = lat2d.shape[1]
nrow = lat2d.shape[0]
res = Ngl.Resources()
res.nglDraw = False
res.nglFrame = False
res.cnFillOn = True
res.cnLinesOn = False
res.cnLineLabelsOn = False
res.lbLabelBarOn = False
res.lbOrientation = "vertical"
res.lbLabelStride = 2
res.cnInfoLabelOn = False
res.cnInfoLabelOrthogonalPosF = 0.04
res.cnInfoLabelString   = ""
res.sfXArray = lon2d[0,:] #scalar field
res.sfYArray = lat2d[:,0]
#res.caXMissingV = -999.
#res.caYMissingV = -999.

res.mpProjection = "LambertConformal"
res.mpLambertParallel1F = 25.
res.mpLambertParallel2F = 40.
res.mpLambertMeridianF  = 110.
res.mpLimitMode       = "Corners"
res.mpLeftCornerLatF  = lat2d[0,0]
res.mpLeftCornerLonF  = lon2d[0,0]
res.mpRightCornerLatF = lat2d[nrow-1,ncol-1]
res.mpRightCornerLonF = lon2d[nrow-1,ncol-1]
res.cnConstFLabelFontHeightF = 0.0
res.tfDoNDCOverlay       = True
res.pmTickMarkDisplayMode = "Always"
res.tmXTOn = False
res.tmYROn = False
#res.tmXBOn = False
#res.tmYLOn = False

res.mpGeophysicalLineThicknessF = 3
res.mpOutlineBoundarySets       = "National"
res.mpOutlineSpecifiers   = ["China:states"]
res.mpDataSetName = "Earth..4"
res.mpDataBaseVersion     = "MediumRes"
res.mpFillOn      = False
res.mpFillAreaSpecifiers  = ["China:states"]

res.tiMainFont    = "helvetica"
res.tiMainOffsetYF= 0
res.tiMainFontHeightF     = 0.02
res.tiMainPosition   = "Left"
res.tiMainFuncCode = "~"

res.mpGridAndLimbOn     = True
res.mpGridLineDashPattern       = 2

#----------------------------

year = '2019'
month = '01'

tvars = ["NO2","SO2","O3","PM25_TOT","PM10","CO"]
tmax  = [ 100,   50, 100,    100,   100, 1000]
tsca  = [46/22.4, 64/22.4, 48/22.4, 1, 1, 28/22.4]

for v in range(6):
    var = tvars[v]
    
    f = Nio.open_file("./cmaq/COMBINE_SA_ACONC_v53_intel_cn27_"+year+month+".nc","r")
    cmaq = np.mean(f.variables[var][15:15+28*24+23,0,:,:],0) * tsca[v]
    
    tmp = np.load("/home/xingj/Documents/covid/obs/obs"+year+month+".npy")
    tmp2 = ma.masked_array(tmp[:,:,0:28,:,v],tmp[:,:,0:28,:,v]==-999.)

    obs = np.mean(tmp2,(2,3))
    
    #----------------------------
    
    
    wks = Ngl.open_wks("png","panel")
    plot = []
    res.cnLevelSelectionMode = "ManualLevels"
    res.cnFillPalette = "BlAqGrYeOrReVi200"
    res.cnMinLevelValF = 0
    res.cnMaxLevelValF =  tmax[v]
    res.cnLevelSpacingF =  tmax[v] / 10.
    res.tiMainString     = "sim: "+var+"-"+year+month
    p = Ngl.contour_map(wks,cmaq,res)
    plot.append(p)
    res.tiMainString     = "obs: "+var+"-"+year+month
    p = Ngl.contour_map(wks,np.transpose(obs),res)
    plot.append(p)
    
    pnlres = Ngl.Resources()
    pnlres.nglFrame = False
    pnlres.txString = var
    pnlres.nglPanelLabelBar    = True
    pnlres.nglPanelLabelBarLabelFontHeightF = 0.01
    Ngl.panel(wks,plot,[1,2],pnlres)
    
    Ngl.frame(wks)
    Ngl.destroy(wks)
    os.system("convert -trim ./panel.png ./panel_"+var+"_"+year+month+".png")
    
