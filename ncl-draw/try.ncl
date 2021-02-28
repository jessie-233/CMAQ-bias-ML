load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl" 

f = asciiread("D:/project/data/pollutant/cmaq/p_sim_daily.npy", (/232, 182, 365, 6/), "float")
var = f(:,:,:,0) 
printVarSummary(var)