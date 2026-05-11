from datasets.utils.progress import Progress
from datasets.utils.interface import Interface
from zipfile import ZipFile
import shutil
import random
import os

class Dataset(Interface):
    
    def __init__(self, zip_path=None, extract_dir=None, dataset_dir=None, objects=None, status=None, width=30, **kwargs):

        """
        Tác dụng:
        - Hàm khởi tạo của class Dataset

        Đầu vào:
        - self: class
        - zip_file: Đường dẫn của nơi lưu file zip dataset
        - extract_dir: Đường dẫn của nơi lưu folder sau unzip của zip_file
        - dataset_dir: Đường dẫn của nơi lưu dataset sau khi xử lý
        - objects: Các đối tượng trong zip_file
        - status: Các trạng thái của objects
        - width: Chiều dài của thanh tiến trình
        - **kwargs: Các thuộc tính khai báo thêm

        Đầu ra:
        - void: Các thuộc tính được khởi tạo tham số ban đầu

        Nguồn: TrinhNhuNhat_06052026.
        """

        if not isinstance(zip_path, str):
            raise TypeError(
                f"zip_path must be str, got {type(zip_path).__name__}"
            )
        
        if not isinstance(extract_dir, str):
            raise TypeError(
                f"extract_dir must be str, got {type(extract_dir).__name__}"
            )
        
        if not isinstance(dataset_dir, str):
            raise TypeError(
                f"dataset_dir must be str, got {type(dataset_dir).__name__}"
            )
        
        if objects is not None:
            if not isinstance(objects, list):
                raise TypeError(
                    f"objects must be list, got {type(objects).__name__}"
                )
            if not all(isinstance(obj, str) for obj in objects):
                raise TypeError(
                    "all elements in objects must be str"
                )
            
        if status is not None:
            if not isinstance(status, list):
                raise TypeError(
                    f"status must be list, got {type(status).__name__}"
                )
            if not all(isinstance(state, str) for state in status):
                raise TypeError(
                    "all elements in status must be str"
                )
            
        if not isinstance(width, int):
            raise TypeError(
                f"width must be int, got {type(width).__name__}"
            )
        
        if width < 0:
            raise ValueError(
                f"width must be int and width > 0, got {width}"
            )
        
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"zip_file not found: {zip_path}"
            )
        
        if not extract_dir.strip():
            raise ValueError(
                "extract_dir cannot be empty or spacewhite"
            )
        
        if not dataset_dir.strip():
            raise ValueError(
                "dataset_dir cannot be empty or spacewhite"
            )
        

        self.progress = Progress( 
            width=width,
            **kwargs
        )
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.dataset_dir = dataset_dir
        self.objects = objects
        self.data_object = [f"data_{obj}" for obj in self.objects]
        self.status = status
        self.total_image = 0
        self.length_image = {}
        self.kwargs = kwargs
        self._images = []
        self.num_classes = 0

    def extract_zip(self, flatten=False, progress=True, default=True, message=False, zip_path=None, extract_path=None, chunk_size=1024 * 64):

        """
        Tác dụng:
        - Hàm giải nén file zip của dataset đầu vào.

        Đầu vào: 
        - self: class
        - flatten: Có gộp tất cả file ở các folder nếu có về chung một không ?
        - progress: Có hiển thị thanh tiến trình không ?
        - end: Có hiện thông báo khi hoàn thành tiến trình không ?
        - chunk_size: Kích thước của từng chunk_size hay đơn vị của tiến trình

        Đầu ra: 
        - void: Trả về folder đã unzip ở self.extract_dir

        Nguồn: TrinhNhuNhat_06052026.
        """

        if not isinstance(flatten, bool):
            raise TypeError(
                f"flatten must be bool, got {type(flatten).__name__}"
            )
        
        if not isinstance(progress, bool):
            raise TypeError(
                f"progress must be bool, got {type(progress).__name__}"
            )
        
        if not isinstance(default, bool):
            raise TypeError(
                f"default must be bool, got {type(default).__name__}"
            )

        if not isinstance(message, bool):
            raise TypeError(
                f"end must be bool, got {type(message).__name__}"
            )
        
        if not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be int, got {type(chunk_size).__name__}"
            )
        
        if not default and not isinstance(zip_path, str):
            raise TypeError(
                f"zip_path must be str when default is False"
            )
        
        if not default and not isinstance(extract_path, str):
            raise TypeError(
                f"extract_path must be str when default is False"
            )
        
        if default and zip_path is not None:
            raise TypeError(
                f"zip_path must be NoneType when default is True"
            )
        
        if default and extract_path is not None:
            raise TypeError(
                f"extract_path must be NoneType when default is True"
            )
        
        if chunk_size <= 0 or chunk_size % 1024 != 0:
            raise ValueError(
                "chunk_size must be positive int and multiple of 1024"
            )
        
        zip_path = self.zip_path if default else zip_path

        with ZipFile(self.zip_path if default else zip_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            total_size = sum(f.file_size for f in file_list if not f.is_dir())

            tracker = self.progress.start(total=total_size) if progress else None
            for file in file_list:

                if file.is_dir() or ".." in file.filename:
                    continue

                output_path = os.path.join(
                    self.extract_dir if default else extract_path,
                    os.path.basename(file.filename) if flatten else file.filename
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with zip_ref.open(file) as source, open(output_path, "wb") as target:
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break

                        target.write(chunk)
                        if tracker:
                            tracker.update(len(chunk))

            if tracker:
                tracker.finish(end=message)


    def _get_folder_by_folder(self, folder=None):

        """
        Tác dụng:
        -Lấy các đường dẫn đến các thư mục

        Đầu vào:
        - self

        Đầu ra:
        - list[str]

        Nguồn: TrinhNhuNhat_07052026.
        """

        if not isinstance(folder, (str, type(None))):
            raise TypeError(
                f"folder must be str or NoneType, got {type(folder).__name__}"
            )
        
        if folder is None:
            return [f"{self.extract_dir}/{obj}" for obj in self.objects]
        
        return [
            f"{folder}/{f}" 
            for f in os.listdir(folder)
            if os.path.isdir(
                f"{folder}/{f}"
            )
        ]
    
    def _get_file_by_folder(self, folder=None):

        if not isinstance(folder, str):
            raise TypeError(
                f"folder must be str, got {type(folder).__name__}"
            )
        
        return [
            f"{folder}/{f}"
            for f in os.listdir(folder)
            if os.path.isfile(
                f"{folder}/{f}"
            )
        ]
    
    def _get_param_by_folder(self, folder=None, param=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff",".webp"]):

        if not isinstance(folder, str):
            raise TypeError(
                f"folder must be str, got {type(folder).__name__}"
            )
        
        if not isinstance(param, list):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if not all(isinstance(p, str) for p in param):
            raise TypeError(
                f"all items in param must be str"
            )
        
        return [
            f"{folder}/{f}"
            for f in os.listdir(folder)
            if f.endswith(
                tuple(param)
            )
        ]
    
    def extract_all(self, progress=True, info=True):

        if not isinstance(progress, bool):
            raise TypeError(
                f"progress must be bool, got {type(progress).__name__}"
            )
        
        if not isinstance(info, bool):
            raise TypeError(
                f"info must be bool, got {type(info).__name__}"
            )

        folders = self._get_folder_by_folder(folder=None)
        for folder in folders:

            files = self._get_file_by_folder(folder=folder)
            total = len(files)

            for idx, file in enumerate(files, start=1): 

                extract_dir = os.path.splitext(file)[0]

                if info:
                    print(
                        f"[INFO {self.progress.now}] "
                        f"Extracting ({idx}/{total}): {file}"
                    )

                self.extract_zip(
                    flatten=False,
                    progress=progress,
                    default=False,
                    message=False,
                    chunk_size=1024 * 64,
                    zip_path=file,
                    extract_path=extract_dir
                )

                if info:
                    print(
                        f"[DONE {self.progress.now}] "
                        f"Extracted to: {extract_dir}"
                    )
        
        if info:
            print(
                f"[INFO {self.progress.now}]: "
                f"All files extracted successfully"
            )

    def clear_folder(self, folder=None, mode="all", exclude=None, progress=True, message=False):

        if not isinstance(folder, str):
            raise TypeError(
                f"folder must be str, got {type(folder).__name__}"
            )
        
        if not isinstance(message, bool):
            raise TypeError(
                f"message must be bool, got {type(message).__name__}"
            )
        
        if not os.path.isdir(folder):
            raise NotADirectoryError(
                f"{folder} is not a valid directory"
            )
        
        if not isinstance(mode, str):
            raise TypeError(
                f"mode must be str, got {type(mode).__name__}"
            )
        
        if mode not in ["all", "file", "folder"]:
            raise ValueError(
                f"mode must is all, file or folder"
            )
        
        if not isinstance(exclude, (type(None), list)):
            raise TypeError(
                f"mode must be str, got {type(mode).__name__}"
            )
        
        if isinstance(exclude, list):
            if not all(isinstance(x, str) for x in exclude):
                raise TypeError(
                    "all items in exclude must be str"
                )
            
        if not isinstance(progress, bool):
            raise TypeError(
                f"progress must be bool, got {type(progress).__name__}"
            )
            
        if exclude is None:
            exclude = []

        items = os.listdir(folder)

        tracker = None

        if progress:
            tracker = Progress(
                desc="Removing",
                unit="item",
                width=30,
            ).start(total=len(items))

        for item in items:
            path = os.path.join(folder, item)
            removed = False

            if item in exclude:
                removed = False
            else:
                if mode in ["all", "file"]:
                    if os.path.isfile(path):
                        os.remove(path)
                        removed = True

                if mode in ["all", "folder"]:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        removed = True

            if tracker:
                tracker.update(1)

        if tracker:
            tracker.finish(end=message)
    
    def _get_path_extract_dir_by_param(self, param=None):

        if not isinstance(param, (str, list)):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if isinstance(param, str) and param not in self.objects:
            raise ValueError(
                f"param must in objects"
            )
        
        param = [param] if isinstance(param, str) else param

        return [
            f"{self.extract_dir}/{obj}"
            for obj in param
        ]

    def _get_all_path_by_param(self, param=None):

        """
        Tác dụng:
        - Lấy tất cả ảnh của các tập dữ liệu

        Đầu vào:
        - self: class
        - param: Là objects muốn lấy dữ liệu

        Đầu ra: 
        - List[str]]: Danh sách tập ảnh

        Nguồn: TrinhNhuNhat_11052026.
        """

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if isinstance(param, str) and param not in self.objects:
            raise ValueError(
                f"param must in objects"
            )

        all_path = []
        folders = self._get_path_extract_dir_by_param(param=self.objects if param is None else param)

        for folder in folders:
            files = self._get_folder_by_folder(folder=folder)
            for file in files:
                file_path = os.path.join(
                    file, os.path.basename(file)
                ).replace("\\", "/")
                all_path.append(file_path)

        return all_path
    
    def _get_all_path_image_by_param(self, param=None):

        """
        Tác dụng:
        - Lấy tất cả danh sách ảnh 

        Đầu vào:
        - self: class

        Đầu ra:
        - List[List[str]]: Danh sách tất cả ảnh trong dataset

        Nguồn: TrinhNhuNhat_11052026
        """

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if isinstance(param, str) and param not in self.objects:
            raise ValueError(
                f"param must in objects"
            )

        images = []
        all_path = self._get_all_path_by_param(param=self.objects if param is None else param)

        for path in all_path:
            image = self._get_param_by_folder(folder=path)
            images.append(image)           
        
        return images
    
    def _merge_index(self, param):

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )

        images = self._get_all_path_image_by_param(param=param)
        label = [images[i][0].split("/")[2] for i in range(len(images))]
        start, point = 0, []
        target = label[0]

        for index in range(len(images)):
            if label[index] != target:
                point.append((start, index - 1))
                start, target = index, label[index]

        point.append((start, len(images) - 1))
        return point
    
    def _merge_list(self, param=None):

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )

        points = self._merge_index(param=param)
        image_before = self._get_all_path_image_by_param(param=param)
        image_after = []

        for point in points:
            start, end = point
            result = []
            for index in range(start, end + 1):
                result += image_before[index]
            image_after.append(result)

        return image_after
    
    def _filter_list(self, param=None, status=None):

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )

        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if not isinstance(status, (str, list, type(None))):
            raise TypeError(
                f"status must be str, got {type(param).__name__}"
            )
        
        if isinstance(status, list) and not all(isinstance(p, str) for p in status):
            raise ValueError(
                f"all items in status must be str"
            )
        
        if isinstance(status, str) and status not in self.status:
            raise ValueError(
                f"invalid status '{status}'. must be one of: {self.status}"
            )
        
        if isinstance(status, list) and not all(state in self.status for state in status):
            raise ValueError(
                f"invalid status list {status}. all values must be in: {self.status}"
            )    

        images_before = self._get_all_path_image_by_param(param=param)
        label = [images_before[i][0].split("/")[3] for i in range(len(images_before))]
        image_after = []

        for index in range(len(images_before)):
            if isinstance(status, str) and label[index] == status:
                image_after.append(images_before[index])
            if isinstance(status, list) and label[index] in status:
                image_after.append(images_before[index])

        return image_after
    
    def _merge_status(self, param=None, status=None):

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if not isinstance(status, (str, list, type(None))):
            raise TypeError(
                f"status must be str, got {type(param).__name__}"
            )
        
        if isinstance(status, list) and not all(isinstance(p, str) for p in status):
            raise ValueError(
                f"all items in status must be str"
            )
        
        if isinstance(status, str) and status not in self.status:
            raise ValueError(
                f"invalid status '{status}'. must be one of: {self.status}"
            )
        
        if isinstance(status, list) and not all(state in self.status for state in status):
            raise ValueError(
                f"invalid status list {status}. all values must be in: {self.status}"
            )

        images_before = self._filter_list(param=param, status=status)
        data = self._index_status(param=param, status=status)
        image_after = [[] for _ in range(len(data))]
        
        for x, value in enumerate(data.values()):
            for y in value:
                image_after[x] += images_before[y]
            
        return image_after
    
    def _index_status(self, param=None, status=None):
        images_before = self._filter_list(param=param, status=status)
        labels = [images_before[i][0].split("/")[3] for i in range(len(images_before))]
        data = {}

        for index, label in enumerate(labels, start=0):
            if label in data:
                data[label].append(index)
            else:
                data[label] = []
                data[label].append(index)

        return data

    def fit(self, param=None, merge=False, status=None):

        """
        Tác dụng:
        - Chia dataset theo điều kiện
        
        Đầu vào: 
        - self: class
        - param: Là objects muốn lấy dữ liệu

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_11052026.
        """

        if not isinstance(param, (str, list, type(None))):
            raise TypeError(
                f"param must be str, got {type(param).__name__}"
            )
        
        if not isinstance(merge, bool):
            raise TypeError(
                f"merge must be str, got {type(merge).__name__}"
            )
        
        if isinstance(param, list) and not all(isinstance(p, str) for p in param):
            raise ValueError(
                f"all items in param must be str"
            )
        
        if isinstance(param, list) and not all(p in self.objects for p in param):
            raise ValueError(
                f"all items in param must in objects"
            )
        
        if not isinstance(status, (str, list, type(None))):
            raise TypeError(
                f"status must be str, got {type(param).__name__}"
            )
        
        if isinstance(status, list) and not all(isinstance(p, str) for p in status):
            raise ValueError(
                f"all items in status must be str"
            )
        
        if isinstance(status, str) and status not in self.status:
            raise ValueError(
                f"invalid status '{status}'. must be one of: {self.status}"
            )
        
        if isinstance(status, list) and not all(state in self.status for state in status):
            raise ValueError(
                f"invalid status list {status}. all values must be in: {self.status}"
            )

        images = (
            (self._merge_status(param=param, status=status) if merge else self._filter_list(param=param, status=status))
            if status is not None
            else (
                self._merge_list(param=param) 
                if merge 
                else self._get_all_path_image_by_param(param=param)
            )
        )

        self.num_classes = len(images)
        
        return [len(image) for image in images]
        

    @property
    def get_params(self):

        """
        Tác dụng:
        - Hàm hiện thị các thông số của thuộc tính trong class

        Đầu vào:
        - self: class

        Đầu ra:
        - void:

        Nguồn: TrinhNhuNhat_07052026.
        """

        items = {}
        if len(self.kwargs):
            for key, value in self.kwargs.items():
                items[key] = value

        infos =  {
            "zip_path": self.zip_path,
            "extract_dir": self.extract_dir,
            "dataset_dir": self.dataset_dir,
            "objects": self.objects,
            "status": self.status,
            "total_image": self.total_image,
            "num_classes": self.num_classes
        }

        infos.update(items)
        return infos


if __name__ == "__main__":
    
    dataset = Dataset(
        zip_path="datasets/raw/Augmented-Resized Image.zip",
        extract_dir="datasets/processed",
        dataset_dir="datasets/dataset",
        objects=["Orange", "Mango", "Grape", "Banana"],
        status=["Rotten", "Fresh", "Formalin-mixed"]
    )

    # dataset.extract_zip(
    #     flatten=False,
    #     progress=True,
    #     default=True,
    #     message=False,
    #     chunk_size=1024 * 64
    # )

    # dataset.extract_all(
    #     progress=True,
    #     info=False
    # )

    # Xóa file zip sau khi giải nén xong
    dataset.clear_folder(
        folder=os.path.dirname(dataset.zip_path),
        mode="file",
        exclude=["__init__.py"]
    )

    # Xóa folder sau khi giải nén xong 
    dataset.clear_folder(
        folder=dataset.extract_dir,
        mode="folder",
        exclude=["__init__.py"]
    )

    # print("Xóa thành công dữ liệu.")

    # print(dataset.fit(merge=False, param=None, status=["Rotten", "Fresh", "Formalin-mixed"]))
    # print(dataset.fit(merge=True, param=None, status=["Rotten", "Fresh"]))
    # print(dataset.fit(merge=True, param=None, status=["Rotten"]))
    # print(dataset.fit(merge=True, param=["Orange", "Mango"], status=None))
    # print(dataset.fit(merge=False, param=["Orange", "Mango"], status=None))
    # print(dataset.fit(merge=True, param=["Orange", "Mango"], status=["Rotten", "Fresh", "Formalin-mixed"]))
    # print(dataset.fit(merge=True, param=None, status=None))
    # # print(dataset.fit(merge=False, param=None, status=None))
    # print(dataset.num_classes)
