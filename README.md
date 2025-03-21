# XÂY DỰNG HỆ THỐNG NHẬN DIỆN CỬ CHỈ TAY HỖ TRỢ NGƯỜI KHUYẾT TẬT GIAO TIẾP

<p align="center">
  <img src="logoDaiNam.png" alt="DaiNam University Logo" width="200"/>
  <img src="LogoAIoTLab.png" alt="AIoTLab Logo" width="170"/>
</p>

[![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

## 🌟 Giới thiệu

Người khuyết tật về nghe và nói gặp nhiều khó khăn trong giao tiếp với cộng đồng. Ngôn ngữ ký hiệu là phương tiện giao tiếp chính của họ, nhưng không phải ai cũng hiểu được. Vì vậy, việc xây dựng một hệ thống nhận diện cử chỉ tay giúp chuyển đổi ngôn ngữ ký hiệu thành văn bản hoặc giọng nói là rất cần thiết.

Với sự phát triển của trí tuệ nhân tạo (AI) và thị giác máy tính, đề tài này tập trung nghiên cứu và ứng dụng các mô hình học sâu để nhận diện cử chỉ tay, giúp người khuyết tật giao tiếp thuận tiện hơn. Hệ thống không chỉ hỗ trợ cá nhân mà còn góp phần tạo ra một môi trường giao tiếp thân thiện, hòa nhập hơn cho cộng đồng.

## 🛠️ Chức năng chính

- **Chuẩn bị dữ liệu:** Tải bộ dữ liệu ASL Alphabet từ Kaggle, giải nén và sắp xếp dữ liệu theo từng lớp ký hiệu.
- **Huấn luyện mô hình:** Hỗ trợ các mô hình CNN, Xception để nhận diện cử chỉ tay. Theo dõi quá trình học qua biểu đồ và ma trận nhầm lẫn.
- **Nhận diện từ ảnh:** Cung cấp một ảnh chứa cử chỉ tay. Hệ thống hiển thị ảnh và dự đoán ký hiệu.
- **Nhận diện từ video:** Tải lên video có chứa cử chỉ tay. Hệ thống tách khung hình và dự đoán từng cử chỉ.
- **Chuyển đổi giọng nói:** Sau khi nhận diện, hệ thống phát âm thanh tương ứng với ký hiệu.
- **Lưu và sử dụng lại mô hình:** Mô hình được lưu lại để sử dụng sau mà không cần huấn luyện lại.

## 🖥️ Công nghệ sử dụng

Hệ thống này được xây dựng với các công nghệ và thư viện sau:

### Trí tuệ nhân tạo (AI) & Học sâu (Deep Learning)
- **TensorFlow & Keras:** Huấn luyện và triển khai mô hình nhận diện cử chỉ tay.
- **Mô hình CNN & Xception:** Dùng để học đặc trưng từ hình ảnh cử chỉ tay.
- **Scikit-learn:** Hỗ trợ tiền xử lý dữ liệu và đánh giá mô hình.

### Thị giác máy tính (Computer Vision)
- **OpenCV:** Xử lý hình ảnh và video để phát hiện và nhận diện cử chỉ tay.

### Phân tích và trực quan hóa dữ liệu
- **NumPy & Pandas:** Xử lý và quản lý dữ liệu.
- **Matplotlib, Seaborn, Plotly:** Vẽ biểu đồ để theo dõi quá trình huấn luyện.
- **Mô hình CNN & Xception:** Hiển thị kết quả huấn luyện và dự đoán trực quan.

### Xử lý ngôn ngữ tự nhiên (NLP) & Tổng hợp giọng nói
- **gTTS (Google Text-to-Speech):** Chuyển đổi văn bản thành giọng nói sau khi nhận diện ký hiệu.

### Lưu và sử dụng lại mô hình
- TensorFlow/Keras hỗ trợ lưu mô hình đã huấn luyện để sử dụng lại mà không cần huấn luyện từ đầu.

## 📚 Dữ liệu sử dụng

Dữ liệu được sử dụng từ Kaggle:
- **Bộ dữ liệu ASL Alphabet:** [Tải tại đây](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- Sau khi tải, giải nén và sắp xếp vào thư mục phù hợp.

## 📚 Các thư viện Python cần thiết

Cài đặt các thư viện bằng lệnh sau:
```sh
pip install opencv-python tensorflow keras numpy pandas scikit-learn matplotlib seaborn plotly gtts
```

## ⚙️ Yêu cầu hệ thống

### Chạy trên Google Colab
Hệ thống có thể chạy trên Google Colab mà không cần cấu hình phức tạp. Chỉ cần tải notebook lên và chạy các ô lệnh theo thứ tự.

#### Hướng dẫn chạy trên Google Colab:
1. **Tải notebook lên Colab:** Mở [Google Colab](https://colab.research.google.com/drive/1168Y2dzgFTMZHrBGYqPZRC3TRRziJFbn?usp=sharing#scrollTo=XMNa-EjUUeR3).
2. **Cài đặt thư viện:** Chạy ô lệnh sau trong Colab để cài đặt thư viện:
   ```python
   !pip install opencv-python tensorflow keras numpy pandas scikit-learn matplotlib seaborn plotly gtts
   ```
3. **Tải dữ liệu ASL Alphabet:**
   ```python
   !kaggle datasets download -d grassknoted/asl-alphabet
   !unzip asl-alphabet.zip -d data/
   ```
4. **Kết nối với GPU (tùy chọn):** Vào `Runtime` > `Change runtime type` > Chọn `GPU` để tăng tốc độ huấn luyện.
5. **Chạy từng ô lệnh theo thứ tự** trong notebook để huấn luyện và kiểm tra mô hình.

### Chạy trên máy tính cá nhân (Visual Studio Code)
Nếu chạy trên Visual Studio Code hoặc môi trường cục bộ:
- **Python 3.7 trở lên**
- **Cài đặt đầy đủ thư viện cần thiết** (xem mục "Các thư viện Python cần thiết")
- **GPU (tùy chọn):** Nếu có GPU, cài đặt CUDA để tăng tốc huấn luyện mô hình

## 🚀 Hướng dẫn cài đặt và chạy chương trình

### 1. Cài đặt môi trường
Nếu chưa có Python, hãy tải và cài đặt Python 3.7 trở lên từ [python.org](https://www.python.org/).

Sau đó, cài đặt các thư viện cần thiết bằng lệnh:
```sh
pip install -r requirements.txt
```
(Nếu không có `requirements.txt`, dùng lệnh `pip install opencv-python tensorflow keras numpy pandas scikit-learn matplotlib seaborn plotly gtts`.)

### 2. Chạy chương trình
#### Trên Google Colab: (Dự đoán bằng ảnh và video)
- **Bước 1: Tải Dataset**
- [Link tải](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Bước 2: Tải thư viện**
  ```python
  !pip install tensorflow opencv-python gtts scikit-learn seaborn
  ```
- **Bước 3: Gắn link dataset**
  ```python
  data='/content/asl_data/asl_alphabet_train/asl_alphabet_train'
  ```
- **Bước 4: Chia ra làm 3 để huấn luyện: 80% train, 10% test, 10% val**
  ```python
  from sklearn.model_selection import train_test_split
  train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 42)
  valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 42)
  ```
- **Bước 5: Xây dựng mô hình Xception, CNN và huấn luyện AI**
  ```python
  base_model = tf.keras.applications.xception.Xception(weights= 'imagenet' ,include_top = False , input_shape = (150,150,3) ,pooling = 'max' )
  model = Sequential([
    BatchNormalization(),
    Dense(256,activation = 'relu'),
    Dropout(.5),
    Dense(29 , activation= 'softmax' )
  ])
  model.compile(Adamax(learning_rate = 0.001) , loss = 'categorical_crossentropy' , metrics= ['accuracy'])
  history = model.fit(
    x= train_generator ,
    validation_data= valid_generator ,
    epochs= 5 , verbose = 1 ,
    validation_steps= None, shuffle= False
  )
  ```
- **Bước 6: Tích hợp Text-to-speech**
  ```python
  !pip install gTTS # install the missing gTTS module
  from gtts import gTTS # import the gTTS module to make it accessible
  from IPython.display import Audio # Import Audio to play the output
  import IPython.display as ipd # import the ipd to be used later
  def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    ipd.display(ipd.Audio("output.mp3", autoplay=True))
  ```
- **Bước 7: Dự đoán hình ảnh**
  ```python
    class_labels = list(train_generator.class_indices.keys())

    def predict_sign(model, img_path):
    # Load và xử lý ảnh
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Chuẩn hóa về [0, 1]
    img = np.expand_dims(img, axis=0)  # Thêm batch dimension
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    label = class_labels[predicted_class]
    return label, confidence, img # Return the image as well
    import matplotlib.pyplot as plt
    test_image = '/content/asl_data/asl_alphabet_test/asl_alphabet_test/A_test.jpg'  # Đổi path tới hình bạn muốn test
    label, confidence, img = predict_sign(model, test_image) # Get the image from the function
    print(f"Dự đoán: {label}, Độ tin cậy: {confidence:.2f}")
    plt.imshow(img[0]) # Display the image using matplotlib
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()
    text_to_speech(label)
  ```
#### Trên Python: (Dự đoán bằng Camera)
- **Huấn luyện mô hình CNN/Xception:** (Do đã có File huấn luyện từ trước có thể bỏ qua bước này)
-  Chạy File train.py
- **Nhận diện cử chỉ từ Camera:**
-  Chạy File predict.py


### Phân chia công việc:
  Phong Ngọc Anh (nhóm trưởng): Phát triển toàn bộ mã nguồn, triển khai dự án, đề xuất cải tiến, kiếm thử, thực hiện video giới thiệu, thuyết trình, Làm Power Point, Github
  
  Bùi Trung Quân: Hỗ trợ làm Power Point, chỉnh sửa Video, Github
  
  Vũ Đức Toàn: Biên soạn tài liệu Overleaf


