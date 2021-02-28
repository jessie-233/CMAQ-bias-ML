load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 

 begin
    f = addfile("D:/project/data/mete/cmaq/GRIDCRO2D_cn27_2017001","r")
    lat2d = f->LAT(0,0,:,:)
    lon2d = f->LON(0,0,:,:)
    nlat = dimsizes(lat2d(:,0))
    nlon = dimsizes(lon2d(0,:))


    pm25 = asciiread("D:/project/data/pollutant/obs/pm25_obs_yearly.csv", (/182,232/), "float")
    pm25@_FillValue = -999.
    pm25@lat2d = lat2d
    pm25@lon2d = lon2d

    wks = gsn_open_wks("X11","D:/project/data/ncl-draw/results/PM25_obs_yearly")
    
    res = True
    res@gsnAddCyclic = False
    res@gsnMaximize = True


    res@cnFillOn = True
    res@cnFillDrawOrder = "Predraw"
    res@cnFillPalette = "WhiteYellowOrangeRed"
    res@cnLinesOn = False
    res@cnLineLabelsOn = False
    res@cnLevelSelectionMode  = "ManualLevels"   ;-- set contour levels manually   
    res@cnMinLevelValF        =  0            ;-- minimum contour level   
    res@cnMaxLevelValF        =  100            ;-- maximum contour level   
    res@cnLevelSpacingF       =  1              ;-- contour level spacing 

    res@lbOrientation          = "Vertical" 
    res@lbTitleOn = True
    res@lbTitleString = "ug/m~S~3~N~"
    res@lbTitleFontHeightF = 0.015
    res@lbTitleOffsetF = -0.05


    res@tfDoNDCOverlay = True
    ;res@sfXArray = lon2d(0,:)
    ;res@sfYArray = lat2d(:,0)
    res@mpProjection = "LambertConformal"
    res@mpLambertParallel1F = 25.
    res@mpLambertParallel2F = 40.
    res@mpLambertMeridianF  = 110.
    res@mpLimitMode  = "Corners"
    res@mpLeftCornerLatF = lat2d(0,nlon-1) 
    res@mpLeftCornerLonF = lon2d(0,nlon-1)   
    res@mpRightCornerLatF = lat2d(nlat-1,0)    
    res@mpRightCornerLonF = lon2d(nlat-1,0) 
    ;res@mpMinLatF  = 35.92
    ;res@mpMaxLatF  = 42.84
    ;res@mpMinLonF  = 112.63
    ;res@mpMaxLonF  = 120.22
    ;res@mpMinLatF  = min(var@lat2d)
    ;res@mpMaxLatF  = max(var@lat2d)
    ;res@mpMinLonF  = min(var@lon2d)
    ;res@mpMaxLonF  = max(var@lon2d)
    res@pmTickMarkDisplayMode = "Always"
    res@tmXTOn = False
    res@tmYROn = False

    ;res@tiMainString  = "PM2.5"     
    ;res@tiMainFontHeightF     =  0.02


    ;res@mpGeophysicalLineThicknessF = 3
    ;res@mpOutlineOn = True
    res@mpOutlineBoundarySets = "NoBoundaries"
    res@mpOutlineSpecifiers = "China:states"
    res@mpDataSetName = "Earth..4"
    res@mpDataBaseVersion = "Ncarg4_1"

    res@mpFillAreaSpecifiers  = (/"water","land"/)
    res@mpSpecifiedFillColors = (/0,0/)
    res@mpAreaMaskingOn = True
    res@mpMaskAreaSpecifiers = "China"
    ;res@mpFillColor = True
    ;res@mpOceanFillColor = 0
    ;res@mpInlandWaterFillColor = 0
    ;res@mpLandFillColor = 0
    res@mpGridAndLimbOn = True
    res@mpGridLineDashPattern = 2

    plot = gsn_csm_contour_map(wks, pm25, res)
 
end


