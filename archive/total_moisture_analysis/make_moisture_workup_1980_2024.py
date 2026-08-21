#!/usr/bin/env python3
import os, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings('ignore', category=FutureWarning)

ANOM_ROOT = Path('../data/era5_anomalies')
SOIL_GLOB = '../data/land/soil_moisture/era5_land_soil_moisture_*.nc'
SST_DIR = Path('/home/k16v981/my_work/data/era5/era5_sst')
EVENT_CSV = Path('../data/wbt_sst_city_runs/city_daily_wbt_JJAS_with_lagged_phases.csv')
PHASE_CSV = Path('../data/sst/roni_dmi_monthly_1950_2025.csv')
ELEVATION_FILE = Path('../data/elevation/GMTED2010_15n060_0250deg.nc')
FIG_DIR = Path('../figures/moisture_workup'); FIG_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_PATH = FIG_DIR / 'moisture_workup_phase_vs_neutral_1980_2024_products.nc'
PNG_PATH = FIG_DIR / 'moisture_workup_phase_vs_neutral_1980_2024.png'
PDF_PATH = FIG_DIR / 'moisture_workup_phase_vs_neutral_1980_2024_manuscript.pdf'

START_YEAR, END_YEAR = 1980, 2024
MONTHS = [6,7,8,9]
DAY_MODE = 'p95'
DATE_COL, CITY_COL, WBT_COL = 'time', 'city', 'wbt_daily_peak'
ENSO_LAG, IOD_LAG = 2, 1
ENSO_POS_THRESH = 0.5; ENSO_NEG_THRESH = -0.5
IOD_POS_THRESH = 0.5; IOD_NEG_THRESH = -0.5
LON_MIN, LON_MAX = 29, 65
LAT_MIN, LAT_MAX = 5, 39
KGKG_TO_GKG = 1000.0
TERRAIN_925_M = 760.0
QUIVER_STRIDE = 6
ROBUST_PCT = 98
COLUMN_KEYS = ['el_nino','la_nina','piod','niod']
COLUMN_TITLES = ['El Niño','La Niña','pIOD','nIOD']
PANEL_LABELS = [f'({chr(97+i)})' for i in range(20)]

ERL_RC = {
    'pdf.fonttype':42,'ps.fonttype':42,'font.family':'sans-serif',
    'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':10,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':8,
    'savefig.transparent':False,
}

def get_lat_lon_names(obj):
    lat = next((n for n in ['latitude','lat','y'] if n in obj.coords or n in obj.dims), None)
    lon = next((n for n in ['longitude','lon','x'] if n in obj.coords or n in obj.dims), None)
    if lat is None or lon is None:
        raise ValueError(f'Could not detect lat/lon: coords={list(obj.coords)}, dims={list(obj.dims)}')
    return lat, lon

def find_time_name(obj):
    for n in ['time','valid_time','date']:
        if n in obj.coords or n in obj.dims:
            return n
    raise ValueError('Could not detect time coordinate')

def subset_ap(da):
    lat, lon = get_lat_lon_names(da)
    da = da.sortby(lat).sortby(lon)
    da = da.sel({lat:slice(LAT_MIN,LAT_MAX), lon:slice(LON_MIN,LON_MAX)})
    if da.sizes.get(lat,0)==0 or da.sizes.get(lon,0)==0:
        raise RuntimeError(f'Empty AP subset for {da.name}: lat={da.sizes.get(lat,0)}, lon={da.sizes.get(lon,0)}')
    return da

def centered_levels(arrays, nlev=21):
    vals=[]
    for a in arrays:
        x=np.asarray(a).ravel(); x=x[np.isfinite(x)]
        if x.size: vals.append(x)
    vals=np.concatenate(vals)
    vmax=np.nanpercentile(np.abs(vals),ROBUST_PCT)
    if not np.isfinite(vmax) or vmax==0: vmax=np.nanmax(np.abs(vals))
    if not np.isfinite(vmax) or vmax==0: vmax=1.0
    return np.linspace(-vmax,vmax,nlev)

def classify_enso(x):
    if pd.isna(x): return np.nan
    if x>=ENSO_POS_THRESH: return 'El Niño'
    if x<=ENSO_NEG_THRESH: return 'La Niña'
    return 'Neutral'

def classify_iod(x):
    if pd.isna(x): return np.nan
    if x>=IOD_POS_THRESH: return 'pIOD'
    if x<=IOD_NEG_THRESH: return 'nIOD'
    return 'Neutral'

def load_monthly_phase_table():
    df=pd.read_csv(PHASE_CSV)
    df['time']=pd.to_datetime(df['time'])
    df['ym']=df['time'].dt.to_period('M')
    df=(df.groupby('ym',as_index=False).agg(RONI=('RONI','mean'),DMI=('DMI','mean')).sort_values('ym').reset_index(drop=True))
    df['time']=df['ym'].dt.to_timestamp()
    df['RONI_lagged']=df['RONI'].shift(ENSO_LAG)
    df['DMI_lagged']=df['DMI'].shift(IOD_LAG)
    df['enso_phase']=df['RONI_lagged'].apply(classify_enso)
    df['iod_phase']=df['DMI_lagged'].apply(classify_iod)
    df=df[(df['time'].dt.year>=START_YEAR)&(df['time'].dt.year<=END_YEAR)&df['time'].dt.month.isin(MONTHS)].copy()
    print('\nJJAS response-month phase counts:')
    print(df['enso_phase'].value_counts()); print(df['iod_phase'].value_counts())
    return df

def load_hhe_dates(phase_df):
    df=pd.read_csv(EVENT_CSV)
    df[DATE_COL]=pd.to_datetime(df[DATE_COL])
    df=df[(df[DATE_COL].dt.year>=START_YEAR)&(df[DATE_COL].dt.year<=END_YEAR)&df[DATE_COL].dt.month.isin(MONTHS)].copy()
    if DAY_MODE=='p95':
        df['threshold']=df.groupby(CITY_COL)[WBT_COL].transform(lambda x:x.quantile(0.95))
        df=df[df[WBT_COL]>=df['threshold']].copy()
    df['ym']=df[DATE_COL].dt.to_period('M')
    lookup=phase_df.set_index('ym')
    df['enso_phase']=df['ym'].map(lookup['enso_phase']); df['iod_phase']=df['ym'].map(lookup['iod_phase'])
    specs=[('el_nino','enso_phase','El Niño'),('la_nina','enso_phase','La Niña'),('enso_neutral','enso_phase','Neutral'),('piod','iod_phase','pIOD'),('niod','iod_phase','nIOD'),('iod_neutral','iod_phase','Neutral')]
    out={}
    for key,col,label in specs:
        sub=df[df[col]==label]
        out[key]=pd.DatetimeIndex(sorted(pd.to_datetime(sub[DATE_COL]).dt.normalize().unique()))
    print('\nHHE date counts:')
    for k,v in out.items(): print(f'  {k:>13s}: {len(v)}')
    return out

def composite_on_dates(da, dates):
    t=find_time_name(da)
    if t!='time': da=da.rename({t:'time'})
    all_dates=pd.DatetimeIndex(pd.to_datetime(da.time.values)).normalize()
    wanted=pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    idx=np.where(np.isin(all_dates.values,wanted.values))[0]
    if len(idx)==0: raise ValueError(f'No matching dates for {da.name}')
    return da.isel(time=idx).mean('time',skipna=True)

def open_anom_field(folder):
    files=sorted((ANOM_ROOT/folder).glob('*.nc'))
    if not files: raise FileNotFoundError(folder)
    ds=xr.open_mfdataset(files,combine='by_coords',parallel=False,engine='h5netcdf',coords='minimal',compat='override')
    if len(ds.data_vars)!=1: raise ValueError(f'{folder}: found vars {list(ds.data_vars)}')
    da=ds[list(ds.data_vars)[0]]
    t=find_time_name(da)
    if t!='time': da=da.rename({t:'time'})
    da=subset_ap(da)
    return da.sel(time=slice(f'{START_YEAR}-01-01',f'{END_YEAR}-12-31'))

def load_atmospheric_fields():
    print('\nOpening atmospheric moisture fields...')
    return {
        'q':open_anom_field('specific_humidity')*KGKG_TO_GKG,
        'u':open_anom_field('u_component_of_wind'),
        'v':open_anom_field('v_component_of_wind'),
        'mfmag':open_anom_field('moisture_flux_mag_925')*KGKG_TO_GKG,
        'mfu':open_anom_field('moisture_flux_u_925')*KGKG_TO_GKG,
        'mfv':open_anom_field('moisture_flux_v_925')*KGKG_TO_GKG,
    }

def load_soil_fields():
    files=sorted(glob.glob(SOIL_GLOB))
    if not files: raise FileNotFoundError(SOIL_GLOB)
    print(f'\nOpening {len(files)} ERA5-Land files...')
    ds=xr.open_mfdataset(files,combine='by_coords',parallel=False,engine='h5netcdf',coords='minimal',compat='override')
    t=find_time_name(ds)
    if t!='time': ds=ds.rename({t:'time'})
    ds=ds.sel(time=slice(f'{START_YEAR}-01-01',f'{END_YEAR}-12-31'))
    lat,lon=get_lat_lon_names(ds); ds=ds.sortby(lat).sortby(lon).sel({lat:slice(LAT_MIN,LAT_MAX),lon:slice(LON_MIN,LON_MAX)})
    if ds.sizes.get(lat,0)==0 or ds.sizes.get(lon,0)==0: raise RuntimeError('Empty soil subset')
    for v in ['swvl1','swvl2']:
        if v not in ds: raise ValueError(f'Missing {v}; found {list(ds.data_vars)}')
    return {'swvl1':ds['swvl1'],'swvl2':ds['swvl2']}

def load_daily_sst():

    files = [
        f"/home/k16v981/my_work/data/era5/era5_sst/"
        f"era5_sst_{year}.nc"
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    missing = [f for f in files if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} SST files. "
            f"First missing: {missing[0]}"
        )

    print(
        f"\nOpening {len(files)} yearly SST files..."
    )

    def preprocess_sst(ds):

        # ERA5 sometimes includes expver in only some files.
        # Collapse/remove it before concatenating yearly datasets.
        if "expver" in ds.dims:
            ds = ds.max("expver", skipna=True)

        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")

        return ds

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        engine="h5netcdf",
        coords="minimal",
        compat="override",
        preprocess=preprocess_sst,
    )

    preferred = [
        "sst",
        "sea_surface_temperature",
    ]

    sst_name = next(
        (
            name for name in preferred
            if name in ds.data_vars
        ),
        None,
    )

    if sst_name is None:

        if len(ds.data_vars) != 1:
            raise ValueError(
                f"Could not identify SST variable. "
                f"Found {list(ds.data_vars)}"
            )

        sst_name = list(ds.data_vars)[0]

    sst = ds[sst_name]

    time_name = find_time_name(sst)

    if time_name != "time":
        sst = sst.rename(
            {time_name: "time"}
        )

    sst = subset_ap(sst)

    sst = sst.sel(
        time=slice(
            f"{START_YEAR}-01-01",
            f"{END_YEAR}-12-31",
        )
    )

    print(
        "Computing daily-mean SST from 6-hourly data..."
    )

    sst_daily = (
        sst
        .resample(time="1D")
        .mean(skipna=True)
    )

    units = str(
        sst.attrs.get("units", "")
    ).lower()

    if units in {"k", "kelvin"}:
        sst_daily = (
            sst_daily - 273.15
        )

        sst_daily.attrs["units"] = "degC"

    return sst_daily

def load_terrain_mask(target_lats,target_lons):
    ds=xr.open_dataset(ELEVATION_FILE)
    var=next((n for n in ['elevation','elev','z','Band1','gmted2010','topography','orog'] if n in ds.data_vars),list(ds.data_vars)[0])
    elev=ds[var].squeeze(drop=True)
    lat,lon=get_lat_lon_names(elev)
    elev=elev.sortby(lat).sortby(lon).sel({lat:slice(LAT_MIN,LAT_MAX),lon:slice(LON_MIN,LON_MAX)})
    if lat!='latitude' or lon!='longitude': elev=elev.rename({lat:'latitude',lon:'longitude'})
    elev=elev.interp(latitude=xr.DataArray(target_lats,dims='latitude'),longitude=xr.DataArray(target_lons,dims='longitude'),method='nearest')
    e=np.asarray(elev.values,dtype=np.float32)
    return np.isfinite(e)&(e>=TERRAIN_925_M)

def build_composites(fields,hhe_dates):
    comps={k:{} for k in COLUMN_KEYS}
    pairs={'el_nino':('el_nino','enso_neutral'),'la_nina':('la_nina','enso_neutral'),'piod':('piod','iod_neutral'),'niod':('niod','iod_neutral')}
    for key in COLUMN_KEYS:
        pkey,nkey=pairs[key]
        print(f'\nBuilding {key} - neutral composites...')
        for name,da in fields.items():
            print(f'  {name}')
            comps[key][name]=(composite_on_dates(da,hhe_dates[pkey])-composite_on_dates(da,hhe_dates[nkey])).load()
    return comps

def save_products(comps):
    # Save separate native-grid groups by encoding each field with field-specific dimensions.
    ds_out=xr.Dataset()
    for phase in COLUMN_KEYS:
        for field,da in comps[phase].items():
            lat,lon=get_lat_lon_names(da)
            latdim=f'{field}_latitude'; londim=f'{field}_longitude'
            arr=da.rename({lat:latdim,lon:londim})
            ds_out[f'{phase}_{field}']=arr
    ds_out.attrs.update({'start_year':START_YEAR,'end_year':END_YEAR,'enso_threshold':ENSO_POS_THRESH,'iod_threshold':IOD_POS_THRESH,'enso_lag_months':ENSO_LAG,'iod_lag_months':IOD_LAG,'day_mode':DAY_MODE})
    ds_out.to_netcdf(PRODUCTS_PATH)
    print(f'\nSaved plotting products: {PRODUCTS_PATH}')

def style_map(ax,labels=False):
    ax.set_extent([LON_MIN,LON_MAX,LAT_MIN,LAT_MAX],crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m',linewidth=0.7,color='0.15',zorder=20)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=0.45,edgecolor='0.20',zorder=20)
    gl=ax.gridlines(draw_labels=labels,linewidth=0.25,color='0.45',alpha=0.4,linestyle='--')
    if labels:
        gl.top_labels=False; gl.right_labels=False
        gl.xlocator=mticker.FixedLocator(np.arange(30,66,5)); gl.ylocator=mticker.FixedLocator(np.arange(5,40,5))
        gl.xlabel_style={'size':7}; gl.ylabel_style={'size':7}

def add_panel_label(ax,label):
    ax.text(0.02,0.98,label,transform=ax.transAxes,ha='left',va='top',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=0.8,pad=1.2),zorder=30)

def add_quiver(ax,u,v,terrain_mask=None):
    lat,lon=get_lat_lon_names(u)
    uu=u.values.copy(); vv=v.values.copy()
    if terrain_mask is not None:
        uu[terrain_mask]=np.nan; vv[terrain_mask]=np.nan
    uu=uu[::QUIVER_STRIDE,::QUIVER_STRIDE]; vv=vv[::QUIVER_STRIDE,::QUIVER_STRIDE]
    xx,yy=np.meshgrid(u[lon].values[::QUIVER_STRIDE],u[lat].values[::QUIVER_STRIDE])
    return ax.quiver(xx,yy,uu,vv,transform=ccrs.PlateCarree(),scale=None,width=0.002,headwidth=3.7,headlength=4.2,minlength=0.05,pivot='middle',color='black',path_effects=[pe.Stroke(linewidth=1.0,foreground='white'),pe.Normal()],zorder=15)

def plot_workup(comps):
    rows=[
        ('q','BrBG',r'925 hPa $q$','g kg$^{-1}$',('u','v'),True),
        ('mfmag','BrBG',r'925 hPa $q\mathbf{v}$','g kg$^{-1}$ m s$^{-1}$',('mfu','mfv'),True),
        ('swvl1','BrBG','Soil moisture layer 1',r'm$^3$ m$^{-3}$',None,False),
        ('swvl2','BrBG','Soil moisture layer 2',r'm$^3$ m$^{-3}$',None,False),
        ('sst','coolwarm','SST',r'$^\circ$C',None,False),
    ]
    levels={f:centered_levels([comps[k][f].values for k in COLUMN_KEYS]) for f,_,_,_,_,_ in rows}
    qref=comps['el_nino']['q']; qlat,qlon=get_lat_lon_names(qref)
    terrain=load_terrain_mask(qref[qlat].values,qref[qlon].values)
    proj=ccrs.PlateCarree()
    with mpl.rc_context(ERL_RC):
        fig,axes=plt.subplots(5,4,figsize=(11.5,12.8),subplot_kw={'projection':proj},constrained_layout=False)
        fig.subplots_adjust(left=0.08,right=0.99,top=0.95,bottom=0.04,wspace=0.045,hspace=0.18)
        for c,t in enumerate(COLUMN_TITLES): axes[0,c].set_title(t,fontsize=10,fontweight='bold')
        pi=0; mappables={}
        for r,(field,cmap,rowlabel,cbarlabel,vectors,is925) in enumerate(rows):
            for c,key in enumerate(COLUMN_KEYS):
                ax=axes[r,c]; style_map(ax,labels=(c==0))
                da=comps[key][field]; lat,lon=get_lat_lon_names(da); vals=da.values.copy()
                if is925: vals[terrain]=np.nan
                cf=ax.contourf(da[lon].values,da[lat].values,vals,levels=levels[field],cmap=cmap,extend='both',transform=proj,zorder=1)
                if c==0: mappables[field]=cf
                if vectors:
                    u,v=vectors; qv=add_quiver(ax,comps[key][u],comps[key][v],terrain_mask=terrain)
                    if c==3 and field=='q': ax.quiverkey(qv,0.49,-0.055,5,'5 m s$^{-1}$',labelpos='E',coordinates='axes',fontproperties={'size':7})
                    if c==3 and field=='mfmag': ax.quiverkey(qv,0.43,-0.055,15,'15 g kg$^{-1}$ m s$^{-1}$',labelpos='E',coordinates='axes',fontproperties={'size':7})
                if is925:
                    ax.contourf(da[lon].values,da[lat].values,terrain.astype(float),levels=[0.5,1.5],colors=['0.88'],transform=proj,zorder=10)
                add_panel_label(ax,PANEL_LABELS[pi]); pi+=1
            axes[r,0].text(-0.25,0.5,rowlabel,transform=axes[r,0].transAxes,rotation=90,ha='center',va='center',fontsize=8.5,fontweight='bold')
            cb=fig.colorbar(mappables[field],ax=axes[r,:],orientation='horizontal',pad=0.045,shrink=0.84,fraction=0.025,aspect=45)
            cb.set_label(cbarlabel); cb.ax.tick_params(labelsize=7)
        fig.savefig(PNG_PATH,dpi=300,bbox_inches='tight')
        fig.savefig(PDF_PATH,dpi=300,bbox_inches='tight',pad_inches=0.02)
        plt.close(fig)
    print(f'Saved PNG: {PNG_PATH}')
    print(f'Saved PDF: {PDF_PATH}')

def main():
    print(f'Moisture workup {START_YEAR}-{END_YEAR}; thresholds +/-0.5; no bootstrap')
    phase_df=load_monthly_phase_table()
    hhe_dates=load_hhe_dates(phase_df)
    atmos=load_atmospheric_fields()
    soil=load_soil_fields()
    sst=load_daily_sst()
    fields={**atmos,**soil,'sst':sst}
    comps=build_composites(fields,hhe_dates)
    save_products(comps)
    plot_workup(comps)
    print('\nDone.')

if __name__=='__main__':
    main()
