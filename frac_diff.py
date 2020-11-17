#Working out fractional derivatives...
#https://en.wikipedia.org/wiki/Fractional_calculus
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import gamma

def is_negative_integer(x:float):
    return x.is_integer() and x < 0

#this function sometimes gives the wrong answer for negative powers
# (eg. integral of x^-1 is ln(x) not 0)
# so the power is limited to non-negative values on the sliders below
#it always gives the right answer as long as power > 0
def analytical_frac_diff(const, power, a):
    '''
    Differentiates [const * x**power] a times, where a is real
    Returns (new_const, new_power)
    '''
    if is_negative_integer(power) and is_negative_integer(power-a):
        #this is actually differentiable and does not yield g(x)=0,
        # but gamma() in the formula would raise a math domain error
        # therefore we treat this case separately
        #note a has to be an integer, so we can fall back on conventional calculus
        if a >= 0:
            #differentiate a times
            for i in range(int(a)):
                const *= power
                power -= 1
        else:
            #integrate a times
            for i in range(-int(a)):
                const /= power+1
                power += 1
        return (const, power)
    elif is_negative_integer(power) or is_negative_integer(power-a):
        return (0.0, 0.0) #NOT ALWAYS THE RIGHT ANSWER WHEN power < 0 !
        #but it does avoid feeding gamma() invalid values below
    else:
        #use the formula from wikipedia for monomial functions
        #d^a/dx^a x^k = gamma(k+1)/gamma(k-a+1) * x^(k-a)
        return (const * gamma(power+1) / gamma(power-a+1), power-a)

def monomial_frac_diff(const, power, a):
    '''
    Differentiates [const * x**power] a times, where a is real
    Returns a numpy vectorized function
    '''
    new_const, new_power = analytical_frac_diff(const, power, a)
    g = lambda x: new_const * x**new_power
    return np.vectorize(g) #return function that can work on numpy arrays

#set up x-values and functions
x_values = np.linspace(0.00001, 5, 1000)
const = 1.0
power = 1.0
order = 0.0
f = np.vectorize(lambda x: const*x**power)
g = monomial_frac_diff(const, power, order)

#set up text rendering
label_font = {'family': 'Calibri',
              'weight': 'normal',
              'size': 12}
def repr_function(const, power):
    '''Represent a monomial function in latex'''
    if const == 0:
        return '0'
    if const == 1 and power == 0:
        return '1'
    const_str = '' if const == 1 else '%.4g' % const
    power_str = '' if power == 0 else 'x' if power == 1 else 'x^{%.4g}' % power
    times = '' if (not const_str) or (not power_str) else r' \, '
    return const_str + times + power_str
def f_label(const, power):
    f_string = repr_function(const, power)
    return f'$f(x)={f_string}$'
def g_label(const, power, order):
    '''Represent g(x) in nice latex'''
    if order == 0:
        d_string = 'f(x)'
    elif order == 1:
        d_string = r'\frac{d}{dx} \, f(x)'
    elif order == -1:
        d_string = r'\int f(x)dx'
    else:
        d_string = r'\frac{d^{%.4g}}{dx^{%.4g}} \, f(x)' % (order, order)
    g_string = repr_function(*analytical_frac_diff(const, power, order))
    return f'$g(x) = {d_string} = {g_string}$'

#set up plots
fig, axs = plt.subplots()
plt.subplots_adjust(bottom=0.25, top=0.93)
f_line, = axs.plot(x_values, f(x_values), color='blue',
                   label = f_label(const, power))
g_line, = axs.plot(x_values, g(x_values), color='green',
                   label = g_label(const, power, order))
axs.set_title('Fractional differentiation')
axs.legend(loc='best', prop=label_font)

#set up sliders
const_limits = ( 0.0, 3.0)
power_limits = ( 0.0, 3.0)
order_limits = (-3.0, 3.0)
axconst = plt.axes([0.15, 0.15, 0.70, 0.03], facecolor='lightgoldenrodyellow')
axpower = plt.axes([0.15, 0.10, 0.70, 0.03], facecolor='lightgoldenrodyellow')
axorder = plt.axes([0.15, 0.05, 0.70, 0.03], facecolor='lightgoldenrodyellow')
s_const = Slider(axconst,'Const c', *const_limits, valinit=const, valstep=0.01)
s_power = Slider(axpower,'Power k', *power_limits, valinit=power, valstep=0.01)
s_order = Slider(axorder,'Order a', *order_limits, valinit=order, valstep=0.01)
def update(val):
    c = s_const.val
    k = s_power.val
    a = s_order.val
    f = np.vectorize(lambda x: c*x**k)
    g = monomial_frac_diff(c, k, a)
    f_line.set_ydata(f(x_values))
    g_line.set_ydata(g(x_values))
    f_line.set_label(f_label(c, k))
    g_line.set_label(g_label(c, k, a))
    axs.legend(loc='best', prop=label_font)
    fig.canvas.draw_idle()
s_order.on_changed(update)
s_power.on_changed(update)
s_const.on_changed(update)

plt.show()
