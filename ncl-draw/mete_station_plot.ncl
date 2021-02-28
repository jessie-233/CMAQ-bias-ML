load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

filepath = "D:/project/data/mete/obs/cma/before_trans.csv"
nrow = 698
ncol = 3
vals = asciiread(filepath, (/nrow,ncol/),"float")
;llfile = np.loadtxt(filepath,dtype="float",delimiter=',',usecols=(1,2))
;print(llfile.shape) #(698, 2) lat lon
lat = vals(:,1)
lon = vals(:,2)

f = addfile("D:/project/data/mete/cmaq/GRIDCRO2D_cn27_2017001","r")
lat2d = f->LAT(0,0,:,:)
lon2d = f->LON(0,0,:,:)
nlat = dimsizes(lat2d(:,0))
nlon = dimsizes(lon2d(0,:))


wks = gsn_open_wks("png", "D:/project/data/ncl-draw/results/mete_stations")
res=True
;res@gsnAddCyclic = False
res@gsnDraw = False 
res@gsnFrame = False
res@gsnMaximize = True 

res@tfDoNDCOverlay = True
res@mpProjection = "LambertConformal"
res@mpLambertParallel1F = 25.
res@mpLambertParallel2F = 40.
res@mpLambertMeridianF  = 110.
res@mpLimitMode  = "Corners"
res@mpLeftCornerLatF = lat2d(0,nlon-1) 
res@mpLeftCornerLonF = lon2d(0,nlon-1)   
res@mpRightCornerLatF = lat2d(nlat-1,0)    
res@mpRightCornerLonF = lon2d(nlat-1,0)
res@tmXTOn = False
res@tmYROn = False


res@mpOutlineBoundarySets = "NoBoundaries"
res@mpOutlineOn = True
res@mpOutlineSpecifiers = "China:states"
res@mpDataSetName = "Earth..4"
res@mpDataBaseVersion = "Ncarg4_1"

res@mpFillAreaSpecifiers  = (/"water","land"/)
res@mpSpecifiedFillColors = (/0,0/)
res@mpAreaMaskingOn = True
res@mpMaskAreaSpecifiers = "China"
res@mpGridAndLimbOn = True
res@mpGridLineDashPattern = 2
;res@mpFillOn = False    
res@mpDataSetName = "Earth..4"   
res@mpDataBaseVersion = "MediumRes"  
res@mpOutlineSpecifiers= "China:States" 
;res@mpMinLatF = 18;36
;res@mpMaxLatF = 54;43
;res@mpMinLonF = 72;113
;res@mpMaxLonF = 137;120
res@pmTickMarkDisplayMode = "Always"
res@tiMainString = "CMA Meteorological Stations" ;China Meteorological Administration

plot = gsn_csm_map(wks,res)

resdot= True
resdot@gsMarkerIndex = 16
resdot@gsMarkerSizeF = 0.005
resdot@gsMarkerColor = "blue"

dumdot = gsn_add_polymarker(wks, plot, lon, lat, resdot)

draw(plot)
frame(wks)
