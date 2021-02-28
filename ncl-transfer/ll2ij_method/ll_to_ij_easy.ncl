load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"


fmap = addfile("D:/project/data/ncl-transfer/ll2ij_method/wrfout_d01_2017-01-31_00_00_00.nc","r")
;;example
;loc3 = wrf_user_ll_to_ij(fmap, 116.37, 39.87, False) - 7 ; lon/lat, return as col/row
;print(loc3) ;136,115

;;北京市
;loc = wrf_user_ll_to_ij(fmap, 115.3, 39.4, False) - 7 ; lon/lat, return as col/row
;print(loc) ;132,113

;loc2 = wrf_user_ll_to_ij(fmap, 117.6, 41.1, False) - 7 ; lon/lat, return as col/row
;print(loc2) ;139,120

;;京津冀
loc = wrf_user_ll_to_ij(fmap, 113.6, 36.1, False) - 7 ; lon/lat, return as col/row
print(loc) ;127,99

loc2 = wrf_user_ll_to_ij(fmap, 120.1, 42.7, False) - 7 ; lon/lat, return as col/row
print(loc2) ;146,128
