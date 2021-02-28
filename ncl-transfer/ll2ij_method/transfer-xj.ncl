load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

begin
	yyyymm	= 202003
	nn2	= systemfunc("wc -l "+yyyymm+".csv|cut -f1 -d' '")
	nn	= toint(nn2) - 1
	print(nn)
	tmp2	= asciiread(yyyymm+".csv",-1,"float")
	;print(tmp2(1010079*9+3:1010079*9+3+9))
	tmp	= reshape(tmp2(3:nn*9+2),(/nn,9/))
	tmp@_FillValue = -999.
	print(tmp(0,:))

	day	= toint(tmp(:,4))

        fmap    = addfile("../wrfout_d01_2017-01-31_00_00_00.nc","r")
        loc     = new((/2,nn/),"integer")
        loc!0   = "ROW"
        loc!1   = "COL"
	opt     = True
        loc     = wrf_user_ll_to_xy(fmap, tmp(:,0), tmp(:,1), opt) - 6 ; lon/lat, return as col/row
	print(loc(:,0))

	data	= new((/182,232,31,4/),"float")
	data!0	= "ROW"
	data!1	= "COL"
	data!2	= "Day"
	data!3	= "conc"
	data@_FillValue = -999.

	do n = 0, nn-1
		print("working on "+n)
		if ((loc(0,n).lt.231).and.(loc(1,n).lt.181).and.(loc(0,n).ge.0).and.(loc(1,n).ge.0)) then
			;print(loc(:,n))
			data(loc(1,n),loc(0,n),day(n)-1,:) = tmp(n,5:8)
		end if
	end do
	system("rm omi"+yyyymm+".nc")
	fo	= addfile("omi"+yyyymm+".nc","c")
	fo->O3	= data(:,:,:,0)
	fo->SO2	= data(:,:,:,1)
	fo->NO2	= data(:,:,:,2)
	fo->PM25 = data(:,:,:,3)
	print(avg(fo->PM25))
end
