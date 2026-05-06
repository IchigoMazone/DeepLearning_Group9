from datasets.utils.progress import Progress
from datasets.utils.interface import Interface
from zipfile import ZipFile
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
                f"width must be int, got {type(int).__name__}"
            )
        
        if width < 0:
            raise ValueError(
                f"width must be int and width > 0, got {width}"
            )

        self.progress = Progress( 
            width=30,
            **kwargs
        )
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.dataset_dir = dataset_dir
        self.objects = objects
        self.data_object = [f"data_{obj}" for obj in self.objects]
        self.status = status
        self.start_object = []
        self.end_object = []
        self.total_image = 0
        self.length_image = {}

    def extract_zip(self, flatten=False, progress=True, end=False, chunk_size=1024 * 64):

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
        
        if not isinstance(end, bool):
            raise TypeError(
                f"end must be str, got {type(end).__name__}"
            )
        
        if not isinstance(chunk_size, int):
            raise TypeError(
                f"chunk_size must be int, got {type(chunk_size).__name__}"
            )
        
        if chunk_size <= 0 or chunk_size % 1024 != 0 or chunk_size / 1024 < 0:
            raise ValueError(
                "chunk_size must be positive int and multiple of 1024"
            )

        with ZipFile(self.zip_path, "r") as zip_ref:
            file_list = zip_ref.infolist()
            total_size = sum(f.file_size for f in file_list if not f.is_dir())

            tracker = self.progress.start(total=total_size) if progress else None
            for file in file_list:

                if file.is_dir() or ".." in file.filename:
                    continue

                output_path = os.path.join(
                    self.extract_dir,
                    os.path.basename(file.filename) if flatten else file.filename
                )

                os.makedirs(output_path, exist_ok=True)

                with zip_ref.open(file) as source, open(output_path, "wb") as target:
                    while True:
                        chunk = source.read(chunk_size)
                        if not chunk:
                            break

                        target.write(chunk)
                        if tracker:
                            tracker.update(len(chunk))

            if tracker:
                tracker.finish(end=end)

    def get_params(self):
        pass


if __name__ == "__main__":
    
    dataset = Dataset(
        zip_path="/content",
        extract_dir="/content",
        dataset_dir="/content",
        objects=["content"],
        status=["status"]
    )
