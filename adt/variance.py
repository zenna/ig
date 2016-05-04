
def powersumavg(bar, series, period, pval=None):
    """
    Returns the power sum average based on the blog post from
    Subliminal Messages.  Use the power sum average to help derive the running
    variance.
    sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/

    Keyword arguments:
    bar     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    period  -- number of values to include in average
    pval    --  previous powersumavg (n - 1) of the series.
    """

    if period < 1:
        raise ValueError("period must be 1 or greater")

    if bar < 0:
        bar = 0

    if pval == None:
        if bar > 0:
            raise ValueError("pval of None invalid when bar > 0")

        pval = 0.0

    newamt = float(series[bar])

    if bar < period:
        result = pval + (newamt * newamt - pval) / (bar + 1.0)

    else:
        oldamt = float(series[bar - period])
        result = pval + (((newamt * newamt) - (oldamt * oldamt)) / period)

    return result

def running_var(bar, series, period, asma, apowsumavg):
    """
    Returns the running variance based on a given time period.
    sources: http://subluminal.wordpress.com/2008/07/31/running-standard-deviations/

    Keyword arguments:
    bar     --  current index or location of the value in the series
    series  --  list or tuple of data to average
    asma    --  current average of the given period
    apowsumavg -- current powersumavg of the given period
    """
    if period < 1:
        raise ValueError("period must be 1 or greater")

    if bar <= 0:
        return 0.0

    if asma == None:
        raise ValueError("asma of None invalid when bar > 0")

    if apowsumavg == None:
        raise ValueError("powsumavg of None invalid when bar > 0")

    windowsize = bar + 1.0
    if windowsize >= period:
        windowsize = period

    return (apowsumavg * windowsize - windowsize * asma * asma) / windowsize
