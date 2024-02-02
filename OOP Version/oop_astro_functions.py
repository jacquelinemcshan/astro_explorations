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


colors=["#bcb056", "#d98556", "#d2617a", "#9c5aa2", "#1b60ad",
        "#20ff0b", "#00dfb4", "#00b2ff", "#0078f6","#6a309a",
        "#65c9ff","#8baaff", "#df75ea", "#ff158d", "#ff0000",
        "#d25dfa", "#ff138a","#fa5916", "#a88d00", "#12a404",
       "#a4eeff", "#96d5ff", "#c7afff", "#fb80c4", "#ff6363"]



class lightcurve_data(lk.collections.LightCurveCollection):
    def __init__(self, lc_data):
        super(lk.collections.LightCurveCollection, self).__init__(lc_data)

        self.stellar_properties=catalog_info(KIC_ID=int(lc_data[0].meta['KEPLERID']))