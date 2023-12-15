import numpy as np

from astropy import units as u

import plotly.graph_objects as go

import lightkurve as lk

import itertools 

from wotan import slide_clip
from wotan import flatten
from wotan import t14

from transitleastsquares import cleaned_array
from transitleastsquares import transitleastsquares
from transitleastsquares import catalog_info
from detecta import detect_peaks




def window_len(data, return_lc=False):
    stellar_properties=catalog_info(KIC_ID=int(data[0].meta['KEPLERID']))
    mass=float(stellar_properties[1])
    radius=float(stellar_properties[4])

    
    stitched=data.stitch()
    quality_index = np.where(stitched['quality'] == 0)[0]
    stitched=stitched[quality_index]
    lc=stitched.flatten(window_length=901).remove_outliers(sigma_lower=np.Inf, sigma_upper=2)
    period_guess = np.linspace(1, 150, 20000)
    pg = lc.to_periodogram(method='bls', period=period_guess, frequency_factor=700)
    period = float(pg.period_at_max_power/u.d)
    tdur = t14(R_s=radius, M_s=mass, P=period, small_planet=True)
    window_leng=3*tdur
    
    if return_lc==False:
        return(window_leng, period) 
    if return_lc==True:
        return(window_leng, period, lc)
    


def tls_peak_periods(tls_results, mult=1.5):

    remove_noise=list(tls_results.power[(tls_results.power > tls_results.stdev)])
    avg=np.average(remove_noise)
    dist=round(len(tls_results.periods)*0.0005)
    
    peak_index=detect_peaks(tls_results.power, mph=mult*avg, mpd=dist,show=False)
    max_peak_index= np.where(tls_results.periods == tls_results.period)[0][0]
    peak_index=peak_index[peak_index!=max_peak_index]

    peak_periods=tls_results.periods[peak_index]


    return(peak_periods)



def harmonic_frac_trace(tls_results, mult=1.5):
    harm_periods=tls_peak_periods(tls_results, mult=1.5)
    
    nones=[None] * len(harm_periods)
    
    x_comp=list(itertools.chain(*zip(harm_periods, harm_periods, nones)))
    x_comp=x_comp[:-1]
    
    y_comp_pattern=[max(tls_results.power)+0.05*max(tls_results.power), -0.05*max(tls_results.power), None]
    
    y_comp=y_comp_pattern*len(harm_periods)
    y_comp=y_comp[:-1]

    harm_frac_lines=go.Scattergl(x=x_comp,y=y_comp, mode='lines',
                                  line=dict(color="#008176",
                                     width=3,
                                     dash="dot"), name='Other Significant <br> Peaks', opacity=0.5, )
    return(harm_frac_lines)



def lightcurve_data(data, window_length=None):
    SAP=[]
    PDCSAP=[]
    norm=[]
    time=[]

    for x in range(0, len(data)-1):
        sap_ind=data[x]['sap_flux'].to_value().tolist()
        pdcsap_ind=data[x]['flux'].to_value().tolist() # source of the flux in the table is PDCSAP FLUX
        norm_ind=data[x].normalize()['flux'].to_value().tolist()
        time_ind=data[x]['time'].to_value('bkjd').tolist()

        SAP.append(sap_ind)
        PDCSAP.append(pdcsap_ind)
        norm.append(norm_ind)
        time.append(time_ind)
    
    stitched=data.stitch()
    quality_index = np.where(stitched['quality'] == 0)[0]
    stitched=stitched[quality_index]
   
    stitched_flux=stitched['pdcsap_flux'].to_value().tolist() 
    stitched_time=stitched['time'].to_value('bkjd').tolist()
    
    if window_length == 'calc':
        window_length=window_len(data)[0]
        period=window_len(data)[1]

        clipped_flux = slide_clip(stitched_time, stitched_flux,
                                  window_length=window_length,
                                  low=3,
                                  high=2,
                                  method='mad',  # mad or std
                                  center='median'  # median or mean
                                  )
        flatten_lc=flatten( 
            stitched_time*u.day,                 # Array of time values
            clipped_flux,                 # Array of flux values
            window_length=window_length, method='biweight', return_trend=False)
        
        detrended_time, detrended_flux= cleaned_array(stitched_time, flatten_lc)
        return(PDCSAP, SAP, norm, time, detrended_flux, detrended_time, stitched, window_length, period)
        
    if window_length == None:
        window_length=0.5
        
        clipped_flux = slide_clip(stitched_time, stitched_flux,
                                  window_length=window_length,
                                  low=3,
                                  high=2,
                                  method='mad',  # mad or std
                                  center='median'  # median or mean
                                  )
        flatten_lc=flatten( 
            stitched_time*u.day,                 # Array of time values
            clipped_flux,                 # Array of flux values
            window_length=window_length, method='biweight', return_trend=False)
        
        detrended_time, detrended_flux= cleaned_array(stitched_time, flatten_lc)
        
        return (PDCSAP, SAP, norm, time, detrended_flux, detrended_time, stitched)



def lightcurve_traces(data, window_length=None):
    upper_range=len(data)-1

    data2plot=lightcurve_data(data, window_length=None)

    SAP_traces={}
    PDCSAP_traces={}
    norm_traces={}
    
    fully_norm_flux_trace=go.Scattergl(x=data2plot[5], y=data2plot[4],  mode='markers',  marker_size=1)

    for x in range(0, upper_range):
        SAP_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[1][x], mode='markers',  marker_size=2)
        PDCSAP_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[0][x], mode='markers',  marker_size=2)
        norm_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[2][x], mode='markers',  marker_size=2)
        
    return (PDCSAP_traces, SAP_traces, norm_traces, fully_norm_flux_trace)



def lightcurve_traces(data, window_length=None):
    upper_range=len(data)-1

    data2plot=lightcurve_data(data, window_length=None)

    SAP_traces={}
    PDCSAP_traces={}
    norm_traces={}
    
    fully_norm_flux_trace=go.Scattergl(x=data2plot[5], y=data2plot[4],  mode='markers',  marker_size=1, marker=dict(color='#0000a7'))

    for x in range(0, upper_range):
        SAP_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[1][x], mode='markers',  marker_size=2)
        PDCSAP_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[0][x], mode='markers',  marker_size=2)
        norm_traces['SAP_trace_' + str(x)]=go.Scattergl(x=data2plot[3][x], name=f"Quarter {data[x].quarter}", y=data2plot[2][x], mode='markers',  marker_size=2)
    return(PDCSAP_traces, SAP_traces, norm_traces, fully_norm_flux_trace)



def lightcurve_plot(data, window_length=None):
    
    traces2plot=lightcurve_traces(data, window_length=None)

    flux_unit=data[0]['flux'].unit.to_string()
    time_unit=data[0]['time'].format

    lc_name=data[0].meta['OBJECT']

    t_ini_limit=float(data[0].meta['TSTART'])
    t_fin_limit=float(data[len(data)-1].meta['TSTOP'])

    PDCSAP_fig_data, SAP_fig_data=list(traces2plot[0].values()), list(traces2plot[1].values())
    norm_fig_data, fully_norm_flux_fig_data=list(traces2plot[2].values()),traces2plot[3]
    
    layout=go.Layout(xaxis=dict(range=[t_ini_limit-20, t_fin_limit+20], autorange=False, zeroline=False))
    
    PDCSAP_fig, SAP_fig=go.Figure(PDCSAP_fig_data, layout), go.Figure(SAP_fig_data, layout)
    norm_fig, fully_norm_flux_fig=go.Figure(norm_fig_data, layout), go.Figure(fully_norm_flux_fig_data, layout)
    
    PDCSAP_fig.update_layout(hovermode=False, font_family="sans-serif", title={
        'text': f"{lc_name}'s PDCSAP Lightcurve",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},  plot_bgcolor='rgba(0,0,0,0)', yaxis_title=f"PDCSAP Flux [{flux_unit}]",
                              xaxis_title=f"Days [{time_unit}]", legend_title="Observation Quarters",
                              xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'), 
                              legend=dict(itemsizing='constant', font=dict(size=10, )))
    
    SAP_fig.update_layout(hovermode=False, font_family="sans-serif", title={
        'text': f"{lc_name}'s SAP Lightcurve",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},  plot_bgcolor='rgba(0,0,0,0)', yaxis_title=f"SAP Flux [{flux_unit}]",
                              xaxis_title=f"Days [{time_unit}]", 
                              xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              legend_title="Observation Quarters",  legend=dict(itemsizing='constant', font=dict(size=10, )))
    
    norm_fig.update_layout(hovermode=False, font_family="sans-serif", title={
        'text': f"{lc_name}'s Normalized Lightcurve",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},  plot_bgcolor='rgba(0,0,0,0)', yaxis_title=f"Normalized Flux",
                              xaxis_title=f"Days [{time_unit}]",xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                                legend_title="Observation Quarters", legend=dict(itemsizing='constant', font=dict(size=10, )))
    
    fully_norm_flux_fig.update_layout(hovermode=False, font_family="sans-serif", title={
       'text': f"{lc_name}'s Normalized and Detrended Lightcurve",
        'y':0.85,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},  plot_bgcolor='rgba(0,0,0,0)', yaxis_title=f"Normalized Flux",
                              xaxis_title=f"Days [{time_unit}]",
        xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
        yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'), 
        legend=dict(itemsizing='constant', font=dict(size=10)))
    
    return(PDCSAP_fig, SAP_fig, norm_fig, fully_norm_flux_fig)



def tls_results(data, max_period=None):
    flux_unit=data[0]['flux'].unit.to_string()
    time_unit=data[0]['time'].format

    if max_period==None or max_period=='calc':
        model_data=lightcurve_data(data, window_length='calc')
        tls_model=transitleastsquares(model_data[5], model_data[4])
        results=tls_model.power(period_min=model_data[8]*0.6, period_max=1.2*model_data[8])
        results.name=data[0].meta['OBJECT']
        results.flux_unit=flux_unit
        results.time_unit=time_unit
        results.stdev=results.depth_mean[0]*np.sqrt(results.in_transit_count)/results.snr
        return(results)
    else:
        model_data=lightcurve_data(data, window_length=None)
        tls_model=transitleastsquares(model_data[5], model_data[4])
        results=tls_model.power(period_min=max_period*0.6, period_max=max_period)
        results.name=data[0].meta['OBJECT']
        results.stdev=results.depth_mean[0]*np.sqrt(results.in_transit_count)/results.snr
        return(results)
    


def tls_plotting(tls_results, mult=1.5, window=0.5):
       
       y_axis_min=-0.05*max(tls_results.power)
       y_axis_max=max(tls_results.power)+0.05*max(tls_results.power)

       folded_mid_point=np.round(tls_results.model_folded_phase[np.where(tls_results.model_folded_model== min(tls_results.model_folded_model))][0],4)
       
       model_phase_max_height=np.average(tls_results.folded_y)+2*np.std(tls_results.folded_y)
       
       model_phase_min_height=min(tls_results.model_folded_model)-1.5*np.std(tls_results.folded_y)

       harm_frac_lines=harmonic_frac_trace(tls_results, mult)
       
       data_tls=go.Scattergl(x=tls_results.periods, y=tls_results.power, mode='lines', 
                             line=dict(color='#0000a7', width=1.5),
                             name="SDE", showlegend=False),
                  
       layout_tls=go.Layout(xaxis=dict(range=[min(tls_results.periods), max(tls_results.periods)], 
                                         autorange=False, zeroline=False),
                                         yaxis=dict(range=[y_axis_min, 
                                                           y_axis_max], 
                                                           autorange=False, zeroline=False)
                                       )
                                       
       tls_fig=go.Figure(data=data_tls, layout=layout_tls)
       tls_fig.add_trace(harm_frac_lines) 

       tls_fig.add_trace(go.Scattergl(x=[tls_results.period, tls_results.period],y=[y_axis_min, y_axis_max], mode='lines',
                                  line=dict(color="#c1272d",
                                     width=5,
                                    ),
                                    name='Maximum Peak', opacity=0.2
                                    )
                                )

                  
                
       tls_fig.update_layout(hovermode=False, font_family="sans-serif", yaxis_title="Spectral Density Estimation (SDE)", xaxis_title="Period [days]",
                      legend_font_size=10,
                      title={
                              'text': f"{tls_results.name}'s TLS Periodogram",
                              'y':0.85,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                             legend={ "x": 1.01,
                                       "y": 0.9,
                                       "xanchor":"left",
                                       "xref": "container",
                                        "yref": "container",
                                          "itemsizing":'constant',
                                          "font":{"size":10}
                             },
                              plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside')
                              )
       
       
       tls_folded=go.Scattergl(x=tls_results.folded_phase, y=tls_results.folded_y,   mode='markers',  marker_size=1.5, marker=dict(color="#c1272d"), name= "Binned Flux")
       tls_model=go.Scattergl(x=tls_results.model_folded_phase, y=tls_results.model_folded_model, name= "TLS Model Flux", mode='lines',
                             line=dict(color='#0000a7', width=2))
      
       tls_folded_data=[tls_folded, tls_model]
      
       layout_tls_folded=go.Layout(xaxis=dict(range=[folded_mid_point*(1-window), folded_mid_point*(1+window)], 
                                         autorange=False, zeroline=False),
                                          yaxis=dict(range=[model_phase_min_height, 
                                                           model_phase_max_height], 
                                                           autorange=False, zeroline=False)
                                         )
       tls_folded_fig=go.Figure(data=tls_folded_data, layout=layout_tls_folded)

       tls_folded_fig.update_layout(hovermode=False, font_family="sans-serif", yaxis_title="Normalized Flux", xaxis_title="Phase",
                      legend_font_size=10,
                      title={
                              'text': f"{tls_results.name}'s Folded Lightcurve",
                              'y':0.85,
                              'x':0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                             legend={ "x": 1.01,
                                       "y": 0.9,
                                       "xanchor":"left",
                                       "xref": "container",
                                        "yref": "container",
                                          "itemsizing":'constant',
                                          "font":{"size":10}
                             },
                              plot_bgcolor='rgba(0,0,0,0)',
                              xaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              yaxis=dict(showline=True, showgrid=False, showticklabels=True, mirror=True,linecolor='black', linewidth=1.5, ticks='outside'),
                              )
       
       return(tls_fig, tls_folded_fig)
