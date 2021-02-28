load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

fmap = addfile("D:/project/data/ncl-transfer/ll2ij_method/wrfout_d01_2017-01-31_00_00_00.nc","r")
;loc = new((/2,698/),"integer")
;loc!0 = "number" 
;loc!1 = "station" 

;对比
tmp = asciiread("D:/project/data/mete/obs/cma/before_trans.csv",(/3,698/),"float") ;是错的，(/.../)行，列
loc = wrf_user_ll_to_ij(fmap, tmp(2,:), tmp(1,:), False) - 7 ; lon/lat, return as col/row
printVarSummary(loc) ;[2] x [698]
;asciiwrite("D:/project/data/mete/cma/after_trans",loc)
;116.366 39.8673
;11621, 3952
print(loc(0,:10))


tmp2 = asciiread("D:/project/data/mete/obs/cma/before_trans.csv",(/698,3/),"float")
loc2 = wrf_user_ll_to_ij(fmap, tmp2(:,2), tmp2(:,1), False) - 7 ; lon/lat, return as col/row
printVarSummary(loc2) ;[2] x [698]
;asciiwrite("D:/project/data/mete/cma/after_trans",loc)
;116.366 39.8673
;11621, 3952
print(loc2(0,:10))