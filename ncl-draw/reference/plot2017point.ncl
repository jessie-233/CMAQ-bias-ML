load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRF_contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;=============================================================================
begin
;=========== parameters ======================================================
  diri          = "/xingjia/dingd/POST/2013-2017/"
  nname		= "2017_2017"
  fname         = diri+"annual_"+nname+".nc"
  
  varhe         = "PM25"
  fmap          = addfile("/home/dingd/CMAQ/scripts/mcip/cn27_2017/GRIDCRO2D_cn27_2017001","r")
;  f             = addfile(fname,"r")
  MaxF          = 60
  MinF          = 0
  SpaceF        = 10

  levels = fspan(MinF, MaxF, (MaxF-MinF)/SpaceF+1)
;  print(levels)
  nlevels = dimsizes(levels)
  
  strs = asciiread("2017annualPM25monitor.txt",-1,"string")
  delim = " ";str_get_tab()
  nfields = str_fields_count(strs(0), delim)
  lat = stringtofloat(str_get_field(strs,2,delim))
  lon = stringtofloat(str_get_field(strs,3,delim))
  monitor = stringtofloat(str_get_field(strs,4,delim))
;-----------------------------------------------------------------------------
;  var           = f->$varhe$
;=========== calulation ======================================================
;  tmp     = ndtooned(var)
;  xmean   = dim_avg(tmp)
;  xdev    = dim_stddev(tmp)
;  xmax    = dim_max(tmp)
;  xmin    = dim_min(tmp)
;  print("mean = "+xmean+"    stddev = "+xdev+"   ( "+xmin+"  --> "+xmax+" )")
  
;=========== plot ============================================================
  wks_type= "png"
  wks     = gsn_open_wks(wks_type,nname+"point")
  clbdft  = "BlAqGrYeOrReVi200"                      ; Colorbar Default
  gsn_define_colormap(wks,clbdft)
  colors  = span_color_rgba(clbdft,nlevels+1)

  res                     = True  ; plot mods desired
  res@gsnDraw             = False ; turn off draw
  res@gsnFrame            = False
  res@gsnSpreadColors     = True  ; use full range of colormap
  res@cnFillOn            = True  ; color plot desired
;  res@cnFillPalette	  = 
  res@cnLinesOn           = False ; turn off contour lines
  res@cnLineLabelsOn      = False ; turn off contour labels

  res@tfDoNDCOverlay              = True
  res@mpGeophysicalLineThicknessF = 2
  res@mpOutlineBoundarySets       = "National"
  res@mpDataSetName         = "Earth..4"   ; This new database contains
                                           ; divisions for other countries.
  res@mpGridAndLimbOn             = True
  res@mpGridLineDashPattern       = 2
  res@tmXTOn                      = False
  res@tmYROn                      = False

;printVarSummary(var)

  res@mpProjection        = "LambertConformal"
  res@mpLambertParallel1F = 25.
  res@mpLambertParallel2F = 40.
  res@mpLambertMeridianF  = 110.
  res@mpLimitMode         = "Corners"
  res@mpLeftCornerLatF    = fmap->LAT(0,0,0,0)
  res@mpLeftCornerLonF    = fmap->LON(0,0,0,0)
  res@mpRightCornerLatF   = fmap->LAT(0,0,181,231)
  res@mpRightCornerLonF   = fmap->LON(0,0,181,231)


  res@tfDoNDCOverlay       = True
  res@pmTickMarkDisplayMode = "Always"         ; turn on tickmarks
 
  res@tmXTOn = False            ; turn off top   labels
  res@tmYROn = False            ; turn off right labels


  res@mpFillOn              = False        ; Turn on map fill
  res@mpDataBaseVersion     = "MediumRes"  ; Medium resolution database
  res@mpFillBoundarySets    = "National"
;  res@mpFillAreaSpecifiers  = (/"China:states"/)
  res@mpOutlineOn           = True         ; Turn on map outlines
  res@mpOutlineSpecifiers   = (/"China:states"/)
                                ; commented 5/17/2007
  res@tfDoNDCOverlay        = True             ; True for 'native' grid
                                ; some WRF are not native


  res@pmTickMarkDisplayMode       = "Always"
  res@lbOrientation       = "Vertical"

  res@cnInfoLabelOn       = True
  res@cnInfoLabelOrthogonalPosF = -0.04
;  res@cnInfoLabelString   = "Max: $ZMX$   Min: $ZMN$   Mean: "+xmean

  res@cnLevelSelectionMode= "ManualLevels"
  res@cnMinLevelValF      = MinF
  res@cnMaxLevelValF      = MaxF
  res@cnLevelSpacingF     = SpaceF
;  res@lbLabelStride       = 10
  res@vpXF                = 0.1

  res@tiMainString        = "PM~B~2.5"
  res@tiMainFontHeightF   = 0.028
  res@tiDeltaF            = 1
  res@gsnLeftString       = "2017"
  res@gsnRightString      = "ug/m~S~3"
  res@gsnLeftStringFontHeightF = 18
  res@gsnRightStringFontHeightF = 18

  res@gsnAddCyclic	  = False
;  plot1    = gsn_csm_contour_map(wks,var(0,0,:,:),res)
  plot1    = gsn_csm_map(wks,res)
  res@gsnLeftString       = ""
  res@gsnRightString      = ""

  num_distinct_markers = nlevels+1        ; number of distinct markers
  lat_new = new((/num_distinct_markers,dimsizes(lat)/),float,-999)
  lon_new = new((/num_distinct_markers,dimsizes(lat)/),float,-999)

  do i = 0, num_distinct_markers-1
    if (i.eq.0) then
      indexes = ind(monitor.lt.levels(0))
    end if
    if (i.eq.num_distinct_markers-1) then
      indexes = ind(monitor.ge.max(levels))
    end if
    if (i.gt.0.and.i.lt.num_distinct_markers-1) then
      indexes = ind(monitor.ge.levels(i-1).and.monitor.lt.levels(i))
    end if
;
; Now that we have the set of indexes whose values fall within 
; the given range, take the corresponding lat/lon values and store
; them, so later we can color this set of markers with the appropriate
; color.
;
    if (.not.any(ismissing(indexes))) then
      npts_range = dimsizes(indexes)   ; # of points in this range.
      lat_new(i,0:npts_range-1) = lat(indexes)
      lon_new(i,0:npts_range-1) = lon(indexes)
    end if
    delete(indexes)            ; Necessary b/c "indexes" may be a different
                               ; size next time.
  end do

  gsres               = True
  base_size = 0.01
  pmid = new(num_distinct_markers,graphic)
  pmid2 = new(num_distinct_markers,graphic)
  do i = 0, num_distinct_markers-1
    if (.not.ismissing(lat_new(i,0)))
      gsres@gsMarkerIndex = 16          ; Use filled dots for markers.
      gsres@gsMarkerColor      = colors(i,:)
;      print(colors(i,:))
      gsres@gsMarkerSizeF      = base_size; * (i+1)/3.
      gsres@gsMarkerThicknessF = 1;*(i+1)
      gsres@gsMarkerOpacityF = 1.0  ; touming
      pmid(i) = gsn_add_polymarker(wks,plot1,lon_new(i,:),lat_new(i,:),gsres)
      gsres@gsMarkerIndex = 4           ; Hollow dots
      gsres@gsMarkerColor      = (/0.00, 0.00, 0.00, 1.00/)
;      gsres@gsMarkerOpacityF = 1.0
      pmid2(i) = gsn_add_polymarker(wks,plot1,lon_new(i,:),lat_new(i,:),gsres)
    end if
  end do

  draw(plot1)
  frame(wks)
  delete(strs)
  delete(lat)
  delete(lon)
  delete(monitor)
  delete(lat_new)
  delete(lon_new)
end
