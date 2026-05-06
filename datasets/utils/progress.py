from datasets.utils.interface import Interface
from datetime import datetime
from tqdm import tqdm
import time
import pytz

class Progress(Interface):

    def __init__(
        self,
        desc="Processing",
        unit="B",
        unit_scale=True,
        timezone="Asia/Ho_Chi_Minh",
        ascii=None,
        mininterval=0,
        dynamic_ncols=False,
        refresh=False,
        width=30,
        **kwargs
    ):
        
        """
        Tác dụng:
        - Hàm khởi tạo của class Progress

        Đầu vào:
        - self: class
        - desc: 
        - unit: 
        - unit_scale: 
        - timezone:
        - ascii: 
        - mininterval: 
        - dynamic_ncols: 
        - refresh:
        - width: 
        - **kwargs:

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """
        
        if not isinstance(desc, str): 
            raise TypeError(
                f"desc must be str, got {type(desc).__name__}"
            )
        
        if not isinstance(unit, str): 
            raise TypeError(
                f"unit must be str, got {type(unit).__name__}"
            )
        
        if not isinstance(unit_scale, bool): 
            raise TypeError(
                f"unit_scale must be bool, got {type(unit_scale).__name__}"
            )
        
        if not isinstance(timezone, str): 
            raise TypeError(
                f"timezone must be str, got {type(timezone).__name__}"
            )
        
        if not isinstance(ascii, (str, type(None))): 
            raise TypeError(
                f"ascii must be str or NoneType, got {type(ascii).__name__}"
            )
        
        if not isinstance(mininterval, (int, float)): 
            raise TypeError(
                f"mininterval must be int or float, got {type(mininterval).__name__}"
            )
        
        if not isinstance(dynamic_ncols, bool): 
            raise TypeError(
                f"dynmic_ncols must be bool, got {type(dynamic_ncols).__name__}"
            )
        
        if not isinstance(refresh, bool):
            raise TypeError(
                f"refresh must be bool, got {type(refresh).__name__}"
            )
        
        if not isinstance(width, (int, type(None))):
            raise TypeError(
                f"width must be int or NoneType, got {type(width).__name__}"
            )
        
        if mininterval < 0: 
            raise ValueError(
                f"mininterval must > 0 and type int, got {mininterval}"
            )
        
        if width is not None and width < 0: 
            raise ValueError(
                f"width must > 0 and type int, got {width}"
            )
        
        self.total = None
        self.pbar = None
        self.start_time = None
        self.bar_format = None
        self.width = None
        self.desc = desc
        self.unit = unit
        self.width = width
        self.unit_scale = unit_scale
        self.mininterval = mininterval
        self.dynamic_ncols = dynamic_ncols
        self.ascii = ascii    
        self.refresh = refresh
        self.location = pytz.timezone(timezone)
        self.kwargs = kwargs
    

    def start (self, total=None, bar_format=None, ascii=None, unit=None):

        """
        Tác dụng:
        - Hàm khởi chạy tiến trình

        Đầu vào:
        - self: class
        - total: 
        - bar_format: 
        - ascii: 
        - unit:

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """

        if bar_format is None:
            if self.width is None:
                bar_format = None
            else:
                bar_format = f"{{l_bar}}{{bar:{self.width}}}{{r_bar}}" 

        if not isinstance(total, int):
            raise TypeError(
                f"total must be int, got {type(total).__name__}"
            )
        
        if not isinstance(bar_format, (str, type(None))): 
            raise TypeError(
                f"bar_format must be str, got {type(bar_format).__name__}"
            )
        
        if not isinstance(ascii, (str, type(None))): 
            raise TypeError(
                f"ascii must be str, got {type(bar_format).__name__}"
            )
        
        if not isinstance(unit, (str, type(None))): 
            raise TypeError(
                f"unit must be str, got {type(unit).__name__}"
            )
        
        if not total > 0: 
            raise ValueError(
                f"total must > 0, got {total}"
            )
        
        self.total = total
        self.bar_format = bar_format
        self.start_time = time.time()
        self.pbar= tqdm(
            total=self.total,
            desc=self.desc,
            unit=unit if unit is not None else self.unit,
            unit_scale=self.unit_scale,
            bar_format=bar_format if bar_format else self.bar_format,
            ascii=ascii if ascii else self.ascii,
            mininterval=self.mininterval,
            dynamic_ncols=self.dynamic_ncols,
            **self.kwargs
        )
        return self

    def update(self, amount):

        """
        Tác dụng:
        - Hàm cập nhật thông số tiến trình theo thời gian thực

        Đầu vào:
        - self: class
        - amount: 

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """

        if self.pbar:
            self.pbar.update(amount)

    def set_metrics(self, refresh=False, **kwargs):

        """
        Tác dụng:
        - Hàm thay đổi các thuộc tính trong tiến trình

        Đầu vào:
        - self: class
        - refresh: 
        - **kwargs: 

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """

        if not isinstance(refresh, bool): 
            raise TypeError(
                f"refresh must be bool, got {type(refresh).__name__}"
            )

        if self.pbar:
            self.pbar.set_postfix(
                **kwargs, 
                refresh=refresh if refresh is not None else self.refresh
            )
    
    def finish(self, desc=None, content=None, end=False):

        """
        Tác dụng:
        - Hàm kết thúc tiến trình

        Đầu vào:
        - self: class
        - desc: 
        - content: 
        - end: 

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """

        if not isinstance(desc, (str, type(None))):  
            raise TypeError(
                f"desc must be str, got {type(desc).__name__}"
            )
        
        if not isinstance(content, (str, type(None))): 
            raise TypeError(
                f"unit must be str, got {type(content).__name__}"
            )
        
        if not isinstance(end, (bool, type(None))): 
            raise TypeError(
                f"unit_scale must be str, got {type(end).__name__}"
            )

        if self.pbar:
            self.pbar.close()
            elapsed = time.time() - self.start_time
            if end:
                print(f"\n{self.desc if not desc else desc} {'done in' if not content else content} {elapsed:.2f}s")

    @property
    def get_params(self):

        """
        Tác dụng:
        - Hàm hiện thị các thông số của thuộc tính trong class

        Đầu vào:
        - self: class

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_06052026.
        """

        items = {}
        if len(self.kwargs):
            for key, value in self.kwargs.items():
                items[key] = value

        infos =  {
            "total": self.total,
            "desc": self.desc,
            "unit": self.unit,
            "unit_scale": self.unit_scale,
            "pbar": self.pbar,
            "start_time": self.start_time,
            "location": self.location,
            "refresh": self.refresh,
            "bar_format": self.bar_format,
            "mininterval": self.mininterval,
            "dynamic_ncols": self.dynamic_ncols,
        }

        infos.update(items)
        return infos

    @property
    def now(self):

        """
        Tác dụng:
        - Hàm lấy thời gian hiện tại

        Đầu vào:
        - self: class

        Đầu ra:
        - datetime:

        Nguồn: TrinhNhuNhat_06052026.
        """

        return datetime.now(self.location).strftime("%d/%m/%Y-%H:%M:%S")

if __name__ == "__main__":

    progress = Progress(colour=None, ascii=None, width=30)
    tracker = progress.start(total=10, ascii=" ░▒▓█", bar_format="{l_bar}{bar:30}")
    for i in range(10):
        tracker.set_metrics(step=tracker.pbar.n, loss=0.5)
        tracker.update(1)
        time.sleep(1)
    tracker.finish()

    tracker = progress.start(total=5)
    for i in range(5):
        tracker.update(1)
        time.sleep(1)
    tracker.finish()