__version__ = '1.0'

# workaround for pyinstaller packing, disabled by default
#import numpy.random.common
#import numpy.random.bounded_integers
#import numpy.random.entropy
#import win32timezone

from functools import partial
from itertools import chain
import math
import os
import pathlib
from random import random

# disable multi-touch emulation
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

import kivy
kivy.require('1.8.0')

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, Point, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, DictProperty
from kivy.uix.bubble import Bubble
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.utils import platform


import numpy

from sfr import SFR 

class Gradient(object):

    @staticmethod
    def horizontal(*args):
        texture = Texture.create(size=(len(args), 1), colorfmt='rgba')
        buf = bytes([ int(v * 255)  for v in chain(*args) ])  # flattens

        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture

    @staticmethod
    def vertical(*args):
        texture = Texture.create(size=(1, len(args)), colorfmt='rgba')
        buf = bytes([ int(v * 255)  for v in chain(*args) ])  # flattens

        texture.blit_buffer(buf, colorfmt='rgba', bufferfmt='ubyte')
        return texture

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    update = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(LoadDialog, self).__init__(**kwargs)
		# Special process for Windows
        if platform == 'win':
            import win32api

            self.ids.spinner.size_hint_max_y = 30
            self.ids.spinner.values = win32api.GetLogicalDriveStrings().split('\000')[:-1]
            self.ids.spinner.values.append(str(pathlib.Path.home()))
            self.ids.spinner.text = self.ids.spinner.values[-1]
            def change_drive(spinner, text):
                self.ids.filechooser.path = text
            self.ids.spinner.bind(text=change_drive)

class MessageBox(Bubble):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    message = StringProperty(None)

class Toast(Bubble):
    message = StringProperty(None)

class RoiSelector(Scatter):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    image = ObjectProperty(None)
    source = StringProperty(None)

    crossline = None
    crossline_label = None
    crossline_label_region = None
    roi = None
    img_offset = (0,0)
    last_image_width = None
    zoom_anchor = (0,0)
    messagebox = None
    touched = False
    mouse_pos_binded = False

    def __init__(self, **kwargs):
        super(RoiSelector, self).__init__(**kwargs)
        self.enable_mouse_move_event(True)
        self.bind(size=self.show)

    def enable_mouse_move_event(self, enabled):
        if enabled == True and self.mouse_pos_binded == False:
            Window.bind(mouse_pos=self.on_mouse_move)
            self.mouse_pos_binded = True
        elif self.mouse_pos_binded == True:
            Window.unbind(mouse_pos=self.on_mouse_move)
            self.mouse_pos_binded = False

    def on_touch_down(self, touch):
        self.touched = True
        if touch.is_mouse_scrolling:
            old_scale = self.scale
            if self.scale == 1:
                self.zoom_anchor = touch.pos
            if touch.button == 'scrolldown' and self.scale < 4:
                self.scale *= 1.25
            elif touch.button == 'scrollup' and self.scale > 1:
                self.scale *= 0.8

            # translate the points from image to self before moving the image
            self.crossline[0].pos = self.pos_image_to_self(self.crossline[0].pos)
            self.crossline[1].pos = self.pos_image_to_self(self.crossline[1].pos)
            self.roi.pos = self.pos_image_to_self(self.roi.pos)
            # new point = (image center - zoom anchor) * (scale - 1) / scale + offset
            self.image.pos = numpy.add( numpy.multiply( numpy.subtract( numpy.divide(self.size,2), self.zoom_anchor), (self.scale-1)/(self.scale)), self.img_offset).tolist()
            # translate the points from self to image after the image moved
            self.crossline[0].pos = self.pos_self_to_image(self.crossline[0].pos)
            self.crossline[1].pos = self.pos_self_to_image(self.crossline[1].pos)
            self.roi.pos = self.pos_self_to_image(self.roi.pos)
            self.update_corssline_label()
            if self.messagebox is not None:
                self.update_messagebox()
        elif touch.button == 'left':
            self.reset_roi()
            touch.grab(self)
            self.roi.pos = self.pos_parent_to_self(touch.pos)
        elif touch.button == 'right' and touch.grab_current is not self:
            self.enable_mouse_move_event(False)
            self.reset_roi()
            self.cancel()

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.roi.size = numpy.subtract(self.pos_parent_to_self(touch.pos), self.roi.pos).tolist()
            self.update_corssline_label()

    def on_touch_up(self, touch):
        if touch.is_mouse_scrolling:
            print(touch.button)
        elif touch.grab_current is self:
            touch.ungrab(self)
            if self.crossline_label.color == [1,1,1,1]:
                self.messagebox = MessageBox(message='Would you like to use this region?', size=(280, 100), cancel=self.reset_roi, load=self.load_roi_image)
                self.parent.add_widget(self.messagebox)
                self.update_messagebox()
            else:
                self.roi.pos = self.pos_parent_to_self(touch.pos)
                self.roi.size = (0, 0)
                self.update_corssline_label()
        self.touched = False

    def on_mouse_move(self, instance, mouse_pos):
        # do not proceed if I'm not displayed <=> If have no parent
        if not self.get_root_window():
            print("Return")
            return

        pos = self.pos_parent_to_self(mouse_pos)

        if self.crossline is None:
            with self.image.canvas:
                # draw the crossline
                Color(1, 0, 0, 0.5)
                self.crossline = [Rectangle(pos=(pos[0], self.image.pos[1]), size=(1, self.height)), Rectangle(pos=(self.image.pos[0], pos[1]), size=(self.width, 1))]
                # draw the ROI
                Color(1, 0, 0, 0.5)
                self.roi = Rectangle(pos=(0, 0), size=(0, 0))
            self.crossline_label = Label(size_hint=(None, None), bold=True, color=(1,1,1,1))
            with self.crossline_label.canvas.before:
                # draw the label background
                Color(0.2, 0.2, 0.2, 0.5)
                self.crossline_label_region = Rectangle(pos=self.crossline_label.pos, size=self.crossline_label.size)
            self.image.add_widget(self.crossline_label)
        elif self.messagebox is None:
            self.crossline[0].pos = (pos[0], self.image.pos[1])
            self.crossline[1].pos = (self.image.pos[0], pos[1])

        # don't update the label when there is a touch event since we've already done it in the touch functions
        if not self.touched:
            self.update_corssline_label()

    def show(self, *args):
        old_img_width = self.image.width
        old_img_offset = self.img_offset
        if self.height*self.image.image_ratio < self.width:
            self.image.size = (self.height*self.image.image_ratio, self.height)
        else:
            self.image.size = (self.width, self.width/self.image.image_ratio)
        self.pos = (-self.width*(self.scale-1)/2, -self.height*(self.scale-1)/2)
        if self.image.width is None:
            ratio = 1
        else:
            ratio = self.image.width/old_img_width
        self.img_offset = ((self.width-self.image.width)/2,(self.height-self.image.height)/2)
        # new point = (old point - old offset) * ratio + new offset
        self.image.pos = numpy.add( numpy.multiply( numpy.subtract(self.image.pos, old_img_offset), ratio), self.img_offset).tolist()
        self.zoom_anchor = numpy.add( numpy.multiply( numpy.subtract(self.zoom_anchor, old_img_offset), ratio), self.img_offset).tolist()
        if self.crossline is not None:
            self.crossline[0].pos = ((self.crossline[0].pos[0] - old_img_offset[0]) * ratio + self.img_offset[0], self.image.pos[1])
            self.crossline[1].pos = (self.image.pos[0], (self.crossline[1].pos[1] - old_img_offset[1]) * ratio + self.img_offset[1])
            self.crossline[0].size = (1, self.height * self.scale)
            self.crossline[1].size = (self.width * self.scale, 1)
            self.roi.pos = numpy.add( numpy.multiply( numpy.subtract(self.roi.pos, old_img_offset), ratio), self.img_offset).tolist()
            self.roi.size = numpy.multiply( self.roi.size, ratio).tolist()
            self.update_corssline_label()
        if self.messagebox is not None:
            def update_messagebox_after_resize(*t):
                self.update_messagebox()
            Clock.schedule_once(update_messagebox_after_resize)
        last_image_width = self.image.width
        return

    def update_messagebox(self):
        pos = self.pos_self_to_parent(self.roi.pos)
        size = numpy.multiply( self.roi.size, self.scale).tolist()
        # find the bottom left point
        start = (min(pos[0], pos[0]+size[0]), min(pos[1], pos[1]+size[1]))
        points = [(pos[0], pos[1])]
        points.append((pos[0], pos[1]+size[1]))
        points.append((pos[0]+size[0], pos[1]+size[1]))
        points.append((pos[0]+size[0], pos[1]))
        for p in points:
            ng = 0
            # test if the point is visible to user
            if p[0] > self.width or p[0] < 0 or p[1] > self.height or p[1] < 0:
                continue
            # test if messagebox will be out of window in vertical direction
            if p[1] == start[1] and p[1]-self.messagebox.height > 0:
                y_dir = 'top'
            elif p[1]+self.messagebox.height < self.height:
                y_dir = 'bottom'
            elif p[1]-self.messagebox.height > 0:
                y_dir = 'top'
            else:
                continue
            if (p[1] == start[1] and y_dir == 'bottom') or (p[1] != start[1] and y_dir != 'bottom'):
                ng += 1
            # test if messagebox will be out of window in horizontal direction
            if ((ng != 0 and p[0] == start[0]) or (ng == 0 and p[0] != start[0])) and p[0]-self.messagebox.width > 0:
                x_dir = 'right'
            elif p[0]+self.messagebox.width < self.width:
                x_dir = 'left'
            elif p[0]-self.messagebox.width > 0:
                x_dir = 'right'
            else:
                continue
            if (p[0] == start[0] and x_dir == 'left') or (p[0] != start[0] and x_dir != 'left'):
                ng += 2
            # setup messagebox
            self.messagebox.x = p[0] if x_dir == 'left' else p[0] - self.messagebox.width
            self.messagebox.y = p[1] if y_dir == 'bottom' else p[1] - self.messagebox.height
            if ng/2 < 1:
                self.messagebox.arrow_pos = x_dir+'_'+y_dir
            else:
                self.messagebox.arrow_pos = y_dir+'_'+x_dir
            if ng == 3 or ng == 0:
                continue
            else:
                break

    def reset_roi(self):
        # TODO: ROI need to be cleared after user back from the SFR viewer. Or, user will see the previous ROI shifted when he pressed the left button.
        if self.messagebox is not None:
            self.parent.remove_widget(self.messagebox)
            self.messagebox = None
            self.roi.size = (0, 0)
            self.update_corssline_label()

    def pos_parent_to_self(self, point):
        # self point = (parent point + (scale - 1) * self center) / scale
        return numpy.divide( numpy.add( point, numpy.multiply( numpy.divide( self.size, 2), self.scale-1)), self.scale).tolist()

    def pos_self_to_parent(self, point):
        # parent point = self point * scale + (1 - sclae) * self center)
        return numpy.add( numpy.multiply( point, self.scale), numpy.multiply( numpy.divide( self.size, 2), 1-self.scale)).tolist()

    def pos_self_to_image(self, point):
        # image point = self point + image position
        return numpy.add( point, self.image.pos).tolist()

    def pos_image_to_self(self, point):
        # self point = image point - image position
        return numpy.subtract( point, self.image.pos).tolist()

    def get_roi_ltrb(self):
        pos = self.pos_image_to_self(self.roi.pos)
        # find the top left and the bottom right points
        start = (min(pos[0], pos[0]+self.roi.size[0]), max(pos[1], pos[1]+self.roi.size[1]))
        end = (max(pos[0], pos[0]+self.roi.size[0]), min(pos[1], pos[1]+self.roi.size[1]))
        roi_start = numpy.multiply( numpy.divide( start, self.image.size), self.image._coreimage.size).tolist()
        roi_end = numpy.multiply( numpy.divide( end, self.image.size), self.image._coreimage.size).tolist()
        # NOTE: (0,0) in python is at the bottom left, but it is at the top left in the image
        roi_start = (round(roi_start[0]), round(self.image._coreimage.size[1]-roi_start[1]))
        roi_end = (round(roi_end[0]), round(self.image._coreimage.size[1]-roi_end[1]))
        return roi_start+roi_end

    def update_corssline_label(self):
        ltrb = self.get_roi_ltrb()
        w = ltrb[2]-ltrb[0]
        h = ltrb[3]-ltrb[1]
        if w == 0 or h == 0:
            self.crossline_label.color = (0.75, 0.75, 0.75, 1)
            self.crossline_label.text = "No ROI"
        elif ltrb[0] < 0 or ltrb[1] < 0 or ltrb[2] > self.image._coreimage.size[0] or ltrb[3] > self.image._coreimage.size[1]:
            self.crossline_label.color = (1, 0, 0, 1)
            self.crossline_label.text = "Invalid ROI"
        else:
            self.crossline_label.color = (1, 1, 1, 1)
            self.crossline_label.text = "Pos: ({0:d}, {1:d})\nSize: {2:d} x {3:d}".format(ltrb[0],ltrb[1],w,h)
        self.crossline_label.texture_update()
        self.crossline_label.pos = (self.crossline[0].pos[0]+1, self.crossline[1].pos[1]+1)
        self.crossline_label.size = (self.crossline_label.texture_size[0] + 20, self.crossline_label.texture_size[1] + 20)
        self.crossline_label_region.pos = self.crossline_label.pos
        self.crossline_label_region.size = self.crossline_label.size

    def load_roi_image(self):
        if self.messagebox is not None:
            self.parent.remove_widget(self.messagebox)
            self.messagebox = None

        # pass back to parent
        self.load(self.image._coreimage.filename, self.get_roi_ltrb())

class SfrViewer(FloatLayout):
    cancel = ObjectProperty(None)
    sfr_dict = DictProperty({})
    export_prefix = StringProperty('')

    crossline = None
    crossline_label = None

    chart_region = None
    x_series = []
    y_series = {}

    data_pt = [0, 0]

    channel = None
    colors = {'L':[[0,0,0,1], [0.3,0.3,0.3,1]],
              'R':[[0.9,0,0,1], [0.7,0,0,1]],
              'G':[[0,0.6,0,1], [0,0.4,0,1]],
              'B':[[0,0,1,1], [0,0,0.8,1]],
              'Line':[[0,0,0,1], [0.7,0.7,0.7,1], [0.5,0.5,0.5,1]],
              'Background':[[0.95,0.95,0.95,1], [1,1,1,1]]}

    charts_exported = False
    widget_initiated = False

    def __init__(self, **kwargs):
        Window.bind(mouse_pos=self.on_mouse_move)
        super(SfrViewer, self).__init__(**kwargs)
        self.bind(pos=self.show, size=self.show)

    def on_mouse_move(self, instance, mouse_pos):
        # do not proceed if I'm not displayed <=> If have no parent
        if not self.get_root_window():
            print("Return")
            return

        # do not proceed if it's not in the region
        if mouse_pos[0] < self.chart_region.pos[0] or mouse_pos[1] < self.chart_region.pos[1] or mouse_pos[0] > self.chart_region.pos[0]+self.chart_region.size[0] or mouse_pos[1] > self.chart_region.pos[1]+self.chart_region.size[1]:
            return

        self.display_selected_data(mouse_pos)

    def display_selected_data(self, pos):
        data_pt = list(self.chart_region.pos)
        # find the closest data point
        for x in self.x_series:
            if abs(x-pos[0]) < abs(data_pt[0]-pos[0]):
                data_pt[0] = x
        idx = self.x_series.index(data_pt[0])
        data_pt[1] = self.y_series[self.channel][idx]

        cy_pxl_series = self.sfr_dict['Cy/Pxl']
        lw_ph_series = self.sfr_dict['LW/PH']
        mtf_series = self.sfr_dict['Channels'][self.channel]['MTF']
        mtf_corr_series = self.sfr_dict['Channels'][self.channel]['Corrected MTF']

        # use interpolation to get the value
        if (idx == 0 and pos[0] < data_pt[0]) or (idx == len(self.x_series)-1 and pos[0] > data_pt[0]):
            cy_pxl = cy_pxl_series[idx]
            lw_ph = lw_ph_series[idx]
            mtf = mtf_series[idx]
            mtf_corr = mtf_corr_series[idx]
        else:
            data_pt[0] = pos[0]
            shift = 1 if pos[0] > data_pt[0] else -1
            slope = (pos[0]-data_pt[0])/(self.x_series[idx+shift]-data_pt[0])
            cy_pxl = cy_pxl_series[idx] + slope * (cy_pxl_series[idx+shift]-cy_pxl_series[idx])
            lw_ph = lw_ph_series[idx] + slope * (lw_ph_series[idx+shift]-lw_ph_series[idx])
            mtf = mtf_series[idx] + slope * (mtf_series[idx+shift]-mtf_series[idx])
            mtf_corr = mtf_corr_series[idx] + slope * (mtf_corr_series[idx+shift]-mtf_corr_series[idx])

        # get the real data point position
        data_pt[1] = self.chart_region.pos[1] + (mtf/max(mtf_series)) * (max(self.y_series[self.channel])-self.chart_region.pos[1])

        if self.crossline is None:
            with self.canvas:
                # draw the crossline
                Color(rgba=self.colors['Line'][2])
                self.crossline = Line(points=[], width=1)
            self.crossline_label = Label(size_hint=(None, None), font_size=18, halign='right', color=self.colors[self.channel][1])
            self.add_widget(self.crossline_label)

        self.crossline.points = [data_pt[0], self.chart_region.pos[1], data_pt[0], data_pt[1], self.chart_region.pos[0], data_pt[1]]
        self.crossline_label.text = 'At {0:0.3f} Cy/Pxl = {1:0.0f} LW/PH:\nMTF = {2:0.3f}    \nMTF(corr) = {3:0.3f}    '.format(cy_pxl, lw_ph, mtf, mtf_corr)
        self.crossline_label.texture_update()
        self.crossline_label.pos = (self.chart_region.pos[0]+self.chart_region.size[0]-self.crossline_label.texture_size[0]-10, self.chart_region.pos[1]+self.chart_region.size[1]-self.crossline_label.texture_size[1]*3.5)
        self.crossline_label.size = self.crossline_label.texture_size

        self.data_pt = data_pt

    def on_touch_up(self, touch):
        if touch.button == 'right':
            self.cancel()
        elif touch.button == 'scrolldown':
            self.switch_to_next_channel(reversed=True)
        else:
            self.switch_to_next_channel()

    def switch_to_next_channel(self, reversed=False):
        channels = list(self.sfr_dict['Channels'].keys())
        idx = channels.index(self.channel)
        if reversed:
            self.channel = channels[idx-1] if idx-1 >= 0 else channels[len(channels)-1]
        else:
            self.channel = channels[idx+1] if idx+1 < len(channels) else channels[0]
        self.show()

    def show(self, *args):
        label = CoreLabel(text="0.0")
        # force refresh to compute things and generate the texture
        label.refresh()
        chart_pos = numpy.add( self.pos, numpy.multiply( numpy.ceil( numpy.divide( label.texture.size, 25)), 25))
        chart_size = numpy.subtract( self.size, numpy.multiply( numpy.ceil( numpy.divide( label.texture.size, 25)), 50)).tolist()
        if self.crossline is not None:
            new_data_pt = numpy.add( numpy.multiply( numpy.subtract( self.data_pt, self.chart_region.pos), numpy.divide( chart_size, self.chart_region.size)), chart_pos)
        self.chart_region = Rectangle(pos=chart_pos, size=chart_size)
        mtf_interval = 0.2
        lw_ph_interval = 500
        max_mtf = 1.0
        for c in self.sfr_dict['Channels'].keys():
            max_mtf = max(max_mtf, max(self.sfr_dict['Channels'][c]['MTF']), max(self.sfr_dict['Channels'][c]['Corrected MTF']))
            if self.channel == None:
                self.channel = c
        max_mtf = round(math.ceil(max_mtf / mtf_interval) * mtf_interval, 1)
        max_lw_ph = max(self.sfr_dict['LW/PH'])
        # re-calculate the lw/ph interval
        ratio = chart_size[0]/chart_size[1]
        for i in [100, 200, 500, 1000, 2000, 5000]:
            if abs(mtf_interval/max_mtf - ratio/math.ceil(max_lw_ph/i)) < abs(mtf_interval/max_mtf - ratio/math.ceil(max_lw_ph/lw_ph_interval)):
                lw_ph_interval = i
        max_lw_ph = round(math.ceil(max_lw_ph / lw_ph_interval) * lw_ph_interval, -2)

        shapes = {}
        self.x_series = pt_x_series = numpy.add( numpy.multiply( self.sfr_dict['LW/PH'], chart_size[0]/max_lw_ph), chart_pos[0]).tolist()
        for c in self.sfr_dict['Channels'].keys():
            mtf_shape = []
            mtf_corr_shape = []
            self.y_series[c] = pt_y_series = numpy.add( numpy.multiply( self.sfr_dict['Channels'][c]['MTF'], chart_size[1]/max_mtf), chart_pos[1]).tolist()
            self.y_series['{0:s}_corr'.format(c)] = pt_y_corr_series = numpy.add( numpy.multiply( self.sfr_dict['Channels'][c]['Corrected MTF'], chart_size[1]/max_mtf), chart_pos[1]).tolist()
            for i in range(len(pt_x_series)):
                mtf_shape.append(pt_x_series[i])
                mtf_shape.append(pt_y_series[i])
                mtf_corr_shape.append(pt_x_series[i])
                mtf_corr_shape.append(pt_y_corr_series[i])

            shapes[c] = [mtf_shape, mtf_corr_shape]

        self.canvas.before.clear()

        with self.canvas.before:
            Color(1,1,1,1)
            Rectangle(texture=Gradient.vertical(self.colors['Background'][0],self.colors['Background'][1]), pos=self.pos, size=self.size)
            Color(rgba=self.colors['Line'][0])
            Line(points=[chart_pos[0], chart_pos[1]+chart_size[1], chart_pos[0], chart_pos[1], chart_pos[0]+chart_size[0], chart_pos[1]], width=1)
            for step in numpy.linspace(0, max_mtf, round(max_mtf/mtf_interval)+1):
                if step != 0:
                    Color(rgba=self.colors['Line'][1])
                    Line(points=[chart_pos[0], chart_pos[1]+step/max_mtf*chart_size[1], chart_pos[0]+chart_size[0], chart_pos[1]+step/max_mtf*chart_size[1]], width=1)
                Color(rgba=self.colors['Line'][0])
                label = CoreLabel(text='{0:0.1f}'.format(step), color=(1, 1, 1, 1))
                label.refresh()
                Rectangle(texture=label.texture, pos=(self.x, chart_pos[1]+step/max_mtf*chart_size[1]-label.texture.size[1]/2), size=label.texture.size)
            for step in numpy.linspace(0, max_lw_ph, round(max_lw_ph/lw_ph_interval)+1):
                if step != 0:
                    Color(rgba=self.colors['Line'][1])
                    Line(points=[chart_pos[0]+step/max_lw_ph*chart_size[0], chart_pos[1], chart_pos[0]+step/max_lw_ph*chart_size[0], chart_pos[1]+chart_size[1]], width=1)
                Color(rgba=self.colors['Line'][0])
                label = CoreLabel(text='{0:0.0f}'.format(step), color=(1, 1, 1, 1))
                label.refresh()
                Rectangle(texture=label.texture, pos=(chart_pos[0]+step/max_lw_ph*chart_size[0]-label.texture.size[0]/2, self.y), size=label.texture.size)

            Color(rgba=self.colors[self.channel][0])
            Line(points=shapes[self.channel][0], width=2)
            label = CoreLabel(text='''[{8:s} Channel]    MTF50 = {0:0.3f} Cy/pxl = {1:0.0f} LW/PH
MTF50P = {4:0.3f} Cy/pxl = {5:0.0f} LW/PH

MTF50(corr) = {2:0.3f} Cy/pxl = {3:0.0f} LW/PH
{6:s} = {7:0.1f} %'''.format(self.sfr_dict['Channels'][self.channel]['MTF50'],
                    self.sfr_dict['Channels'][self.channel]['MTF50']*self.sfr_dict['Size'][0 if self.sfr_dict['Orientation'] == 'Horizontal' else 1]*2,
                    self.sfr_dict['Channels'][self.channel]['Corrected MTF50'],
                    self.sfr_dict['Channels'][self.channel]['Corrected MTF50']*self.sfr_dict['Size'][0 if self.sfr_dict['Orientation'] == 'Horizontal' else 1]*2,
                    self.sfr_dict['Channels'][self.channel]['MTF50P'],
                    self.sfr_dict['Channels'][self.channel]['MTF50P']*self.sfr_dict['Size'][0 if self.sfr_dict['Orientation'] == 'Horizontal' else 1]*2,
                    'Undersharpening' if self.sfr_dict['Channels'][self.channel]['Sharpening'] < 0 else 'Oversharpening',
                    abs(self.sfr_dict['Channels'][self.channel]['Sharpening'])*100,
                    self.channel),
                font_size=18, halign='right', color=(1, 1, 1, 1))
            label.refresh()
            Rectangle(texture=label.texture, pos=(chart_pos[0]+chart_size[0]-label.texture.size[0]-10,chart_pos[1]+chart_size[1]-label.texture.size[1]-10), size=label.texture.size)

            Color(rgba=self.colors[self.channel][1])
            Line(points=shapes[self.channel][1], width=1)

        if self.crossline is not None:
            self.crossline_label.pos = (self.chart_region.pos[0]+self.chart_region.size[0]-self.crossline_label.texture_size[0]-10, self.chart_region.pos[1]+self.chart_region.size[1]-self.crossline_label.texture_size[1]*3.5)
            self.crossline_label.color = self.colors[self.channel][1]
            self.display_selected_data(new_data_pt)

        if not self.widget_initiated:
            self.widget_initiated = True
        elif not self.charts_exported:
            self.export_to_png('{0:s}_{1:s}_MTF.png'.format(self.export_prefix,self.channel))
            channels = list(self.sfr_dict['Channels'].keys())
            if channels.index(self.channel) == len(channels)-1:
                self.charts_exported = True
            self.switch_to_next_channel()

class Workspace(FloatLayout):
    roi_selector = None

    def on_touch_down(self, touch):
        if self.roi_selector is None:
            touch.grab(self)
        else:
            return super(Workspace, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            if touch.button == 'left':
                self.show_load_dialog()
        else:
            return super(Workspace, self).on_touch_up(touch)

    def show_load_dialog(self):
        self._popup = Popup(title='Load image file', title_size='24sp', size_hint=(0.9, 0.9))
        self._popup.content = LoadDialog(load=self.load_image_to_selector, cancel=self._popup.dismiss, update=self.update_dialog_image)
        self._popup.open()

    def load_image_to_selector(self, file_path):
        if os.path.isfile(file_path):
            self.roi_selector = RoiSelector(source=file_path, load=self.get_sfr_from_roi, cancel=self.close_roi_selector)
            self._popup.dismiss()
            self.add_widget(self.roi_selector)

    def update_dialog_image(self, file_path, image):
        self._popup.title=file_path
        if os.path.isfile(file_path):
            image.source = file_path
            image.color = [1,1,1,1]

    def get_sfr_from_roi(self, image_path, image_roi):
        toast = Toast(message='Calculating MTF ...', size=(140, 50), pos_hint={'center_x': .5, 'center_y': .5})
        self.add_widget(toast)
        def calculate_sfr(*t):
            os.chdir(os.path.dirname(image_path))
            if not os.path.isdir('Results'):
                os.mkdir('Results')
            os.chdir('Results')
            outputs = SFR(image_path, image_roi, oversampling_rate=4).calculate(export_csv='{0:s}_summary.csv'.format(os.path.basename(os.path.splitext(image_path)[0])))
            self.remove_widget(toast)
            self.roi_selector.image.export_to_png('{0:s}_MTF_ROI.png'.format(os.path.basename(os.path.splitext(image_path)[0])))
            self._popup = Popup(title='Results of {0:s}'.format(os.path.basename(image_path)), title_size='24', size_hint=(0.9, 0.9))
            self.roi_selector.enable_mouse_move_event(False)
            self._popup.content = SfrViewer(sfr_dict=outputs, cancel=self._popup.dismiss, export_prefix=os.path.basename(os.path.splitext(image_path)[0]))
            def dismiss_callback(instance):
                self.roi_selector.enable_mouse_move_event(True)
                Window.unbind(mouse_pos=self._popup.content.on_mouse_move)
            self._popup.bind(on_dismiss=dismiss_callback)
            self._popup.open()
        Clock.schedule_once(calculate_sfr)

    def close_roi_selector(self):
        self.remove_widget(self.roi_selector)
        self.roi_selector = None

class SfrApp(App):
    title = 'SFR Calculator'

    def build(self):
        self.workspace = Workspace()
        return self.workspace

if __name__ == '__main__':
    SfrApp().run()
