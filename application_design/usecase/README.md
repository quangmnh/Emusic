Use case description:
--
### Login

| Use case ID: | EMS_UC_1 |
| --- | --- |
| Use case name: | Login |
| Description | Use case dùng để giúp user login vào hệ thống bằng Spotify credentials, giúp user sử dụng các tiện ích được cung cấp bởi hệ thống. |
| Primary actor | User |
| Secondary actor | Spotify |
| Preconditions | User vào được hệ thống và có tài khoản Spotify hợp lệ (Premium is recommended) |
| Postconditions | Hệ thống hiển thị màn hình phù hợp |
| Mainflow | 1. User nhấn vào nút “LOGIN WITH SPOTIFY” |
|  | 2. System hiển thị màn hình login thông qua request tới Spotify |
|  | 3. User nhập thông tin tài khoản Spotify và nhấn “LOGIN” |
|  | 4. System gửi request login với thông tin của user về Spotify và nhận response |
|  | 5. System hiển thị màn hình chính của ứng dụng. |
| Alternative flows: | 4a. Thông tin của user không hợp lệ: |
|  | - System hiển thị thông báo yêu cầu người dùng nhập lại thông tin |
|  | - Use case tiếp tục main flow tại bước 3 |
|  | 4b. Thông tin của user không authenticate bởi Spotify được: |
|  | - System hiển thị thông báo yêu cầu người dùng nhập lại thông tin |
|  | - Use case tiếp tục main flow tại bước 3 |
|  | 2a. 4c. System không thể gửi request tới Spotify hoặc không nhận được response: |
|  | - System hiển thị lỗi hệ thống |
|  | - Use case kết thúc tại đây |

### Search

| Use case ID: | EMS_UC_2 |
| --- | --- |
| Use case name: | Search |
| Description | Use case dùng để giúp user tìm kiếm các track, playlist yêu thích phục vụ cho việc nghe nhạc |
| Primary actor | User |
| Secondary actor | Spotify |
| Preconditions | User đã đăng nhập thành công và đang ở màn hình chính của ứng dụng |
| Postconditions | Hệ thống hiển thị danh sách kết quả tìm kiếm dựa trên từ khoá user cung cấp |
| Mainflow | 1. User nhấn vào nút “SEARCH” trên màn hình chính |
|  | 2. System hiển thị màn hình với thanh tìm kiếm |
|  | 3. User nhập từ khoá vào thanh tìm kiếm |
|  | 4. System gửi request search cho Spotify với các từ khoá đã nhập của user nhận response |
|  | 5. System hiển thị danh sách kết quả tìm kiếm từ response nhận được từ Spotify |
| Alternative flows: | 4a. Từ khoá tìm kiếm của user không hợp lệ |
|  | - System hiển thị thông báo yêu cầu người dùng nhập lại thông tin |
|  | - Use case tiếp tục main flow tại bước 3 |
|  | 4b. System không thể gửi request tới Spotify hoặc không nhận được response: |
|  | - System hiển thị lỗi hệ thống |
|  | - Use case kết thúc tại đây |

### Create Emotion Based Playlist

| Use case ID: | EMS_UC_3 |
| --- | --- |
| Use case name: | Create Emotion-based playlits |
| Description | Use case dùng để giúp user tạo các playlist tương ứng với các cảm xúc của user (ví dụ playlist nghe khi buồn, playlist nghe khi vui, etc.) bằng việc lấy thông tin về các từ khoá liên quan đến các track mà user muốn nghe.  |
| Primary actor | User |
| Secondary actor | Spotify |
| Preconditions | User vào hiện đang ở trang tạo playlist theo cảm xúc |
| Postconditions | Hệ thống tạo playlist phù hợp với cảm xúc của user |
| Mainflow | 1. System hiển thị survey cho việc tạo playlist phù hợp với cảm xúc |
|  | 2. User điền các từ khoá vào survey và xác nhận |
|  | 3. System thực hiện gửi request tìm kiếm các track phù hợp với survey cho Spotify và nhận response |
|  | 4. System hiển thị các track được đề xuất từ response được trả về |
|  | 5. User chấp nhận tạo playlist phù hợp với cảm xúc với các track đã được đề xuất |
|  | 6. System tiến hành tạo playlist dựa trên các track đã được User chấp nhận |
| Alternative flows: | 3a. Thông tin của user không hợp lệ: |
|  | - System hiển thị thông báo yêu cầu người dùng nhập lại thông tin |
|  | - Use case tiếp tục main flow tại bước 3 |
|  | 3b. System không thể gửi request hay không thể nhận được response: |
|  | - System hiển thị lỗi hệ thống |
|  | - Use case kết thúc tại đây |
|  | 5a. User thực hiện xoá một số track đã được đề xuất |
|  | - System tiến hành xoá bỏ các track đó |
|  | - Use case tiếp tục với main flow tại bước 5 |

### Select emotion-based playlist

| Use case ID: | EMS_UC_4 |
| --- | --- |
| Use case name: | Select Emotion-based playlits |
| Description | Use case dùng để giúp user chọn playlist tương ứng với các cảm xúc hiện tại của user (ví dụ nghe playlist buồn với cảm xúc hiện tại là buồn, etc.) bằng việc nhận diện cảm xúc khuôn mặt hiện tại của User thông qua FERModel |
| Primary actor | User |
| Secondary actor | FERModel |
| Preconditions | User lựa chọn chế độ lựa chọn playlist chơi nhạc dựa trên cảm xúc |
| Postconditions | Hệ thống chọn và phát playlist phù hợp với cảm xúc của user |
| Mainflow | 1. System tiến hành sử dụng camera để quay khuôn mặt user |
|  | 2. System gửi khuôn mặt đã quay làm input cho FERModel |
|  | 3. FERModel tiến hành dự đoán cảm xúc khuôn mặt dựa trên input được cung cấp |
|  | 4. FERModel gửi trả kết quả dự đoán về cho System |
|  | 5 System tiến hành phát playlist tương ứng với kết quả dự đoán cảm xúc nhận từ FERModel |
| Alternative flows: | 3a. FERModel không thể dự đoán cảm xúc dựa trên input được cung cấp |
|  | - System hiển thị thông báo lỗi, gợi ý người dùng tiến hành lại quy trình |
|  | - Use case kết thúc tại đây |

### Listen to playback

| Use case ID: | EMS_UC_5 |
| --- | --- |
| Use case name: | Listen to playback |
| Description | Use case dùng để giúp User thực hiện việc phát các track hay playlist trên player của System |
| Primary actor | User |
| Preconditions | User ở màn hình hiển thị danh sách các track hoặc playlist |
| Postconditions | Hệ thống phát track hoặc playlist do User chọn |
| Mainflow | 1. User click chọn track/playlist/album |
|  | 2. System lấy thông tin của track/playlist/album |
|  | 3. System gửi thông tin cho player để player phát |
|  | 4. Player phát track/playlist/album tương ứng với thông tin đã được gửi |
| Alternative flows: | 4a. System gửi thông tin cho Player không thành công |
|  | - System hiển thị thông báo lỗi |
|  | - Use case kết thúc tại đây |

### Edit playlist

| Use case ID: | EMS_UC_6 |
| --- | --- |
| Use case name: | Edit playlist |
| Description | Use case dùng để giúp User chỉnh sửa playlist sao cho phù hợp với sở thích |
| Primary actor | User |
| Secondary actor | Spotify |
| Preconditions | User ở màn hình hiển thị danh sách các track của playlist nào đó/ danh sách các track |
| Postconditions | Hệ thống tiến hành cập nhật playlist phù hợp với nhu cầu của User |
| Mainflow | 1. User click chọn option “REMOVE FROM PLAYLIST” hay “ADD TO PLAYLIST” của một track. |
|  | 2. System lấy thông tin của track/playlist |
|  | 3. System gửi thông tin cho Spotify để thực hiện việc remove/add track vào playlist  |
|  | 4. Spotify tiến hành thực hiện cập nhật playlist theo yêu cầu và gửi response |
|  | 5. System nhận response và hiển thị kết quả  |
| Alternative flows: | 3a. System gửi thông tin cho Spotify không thành công |
|  | - System hiển thị lỗi hệ thống |
|  | - Use case kết thúc tại đây |
|  | 5a. System không nhận được response từ Spotify |
|  | - System hiển thị lỗi hệ thống |
|  | - Use case kết thúc tại đây |
