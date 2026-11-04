# Danh Mục Dataset Công Khai Cho Phần Đánh Giá (eKYC CCCD)

Tổng hợp các bộ dữ liệu **đã public** có thể dùng ngay cho Chương 4 (Thực nghiệm
& đánh giá) của luận văn, ghép theo từng module của hệ thống trong
[RESEARCH_DIRECTION.vi.md](RESEARCH_DIRECTION.vi.md).

Ký hiệu mức truy cập:
- **🟢 Mở** — tải trực tiếp, không cần xin phép.
- **🟡 Thỏa thuận** — phải ký release agreement / gửi email cho tác giả (thường
  duyệt vài ngày, miễn phí cho mục đích học thuật).
- **🔴 Hạn chế** — có thể đã ngừng phát tán hoặc phải qua quy trình phức tạp.

---

## 1. Liveness thụ động / chống giả mạo khuôn mặt (MiniFASNetV2)

Module cần đánh giá: [fasnet.py](ai/app/services/fasnet.py),
[liveness_service.py](ai/app/services/liveness_service.py) → chỉ số APCER/BPCER.

| Dataset | Nội dung | Mức truy cập | Ghi chú sử dụng |
| --- | --- | --- | --- |
| **Replay-Attack (Idiap)** | 1300 clip video, 50 chủ thể, tấn công ảnh in + replay màn hình, nhiều điều kiện sáng và khoảng cách | 🟢 [idiap.ch/en/dataset/replay-attack](https://www.idiap.ch/en/dataset/replay-attack), mirror [zenodo.org/records/4593128](https://zenodo.org/records/4593128) | Chuẩn "greeting card" cho luận văn: nhỏ, dễ reproducible, có protocol công bố |
| **MSU-MFSD (v1/v2)** | Ảnh in + replay từ điện thoại / tablet quay lại bằng webcam | 🟡 [Release Agreement (PDF)](http://biometrics.cse.msu.edu/Publications/Databases/MSUMobileFaceSpoofing/Agreement.pdf); bản đã tách frame: [Kaggle](https://www.kaggle.com/datasets/minhtranv/msu-mfsd-processed-into-frames) | Khớp nhất với kịch bản webcam của bạn; trang tác giả: [sites.google.com/site/huhanhomepage/datasetcode](https://sites.google.com/site/huhanhomepage/datasetcode) |
| **CASIA-FASD** | 50 chủ thể, 600 video (3 thật + 9 giả/chủ thể), 3 loại tấn công (photo cong, photo khoét mắt, video replay), 3 mức chất lượng | 🟡 Chính thống: [trang CSDL CBSR](http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp) + [Agreement PDF](http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf) — ký, upload/email, nhận account tải trong 48 giờ | Benchmark kinh điển, được trích dẫn nhiều nhất; bản crops 73 MB chỉ là frame cắt sẵn, không có video |
| **OULU-NPU** | 4950 video, 55 chủ thể, 3 thiết bị × 2 môi trường × 4 điều kiện sáng (in + replay) | 🟡 [sites.google.com/site/oulunpudatabase](https://sites.google.com/site/oulunpudatabase/) | Mạnh nhất để nói về **độ bền theo biến đổi thực tế** (device/illumination) |
| **ROSE-Youtu** | ~4200 clip, in / replay / mask giấy 3D, 5 thiết bị di động, đa mức sáng | 🟢 [rose1.ntu.edu.sg/dataset/faceLivenessDetection](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/) | Có sẵn nhãn phân theo loại tấn công → rất tiện cho bảng kết quả theo từng attack |
| **SiW / SiW-M (MSU)** | SiW: lớn, in + replay trên di động. SiW-M: 493 chủ thể, 13–14 loại tấn công (mask, hóa trang, vật liệu) | 🟡 [cvlab.cse.msu.edu/project-face-anti.html](http://cvlab.cse.msu.edu/project-face-anti.html), [downloads](http://cvlab.cse.msu.edu/category/downloads.html) | Dùng để **chứng minh hạn chế** đã ghi trong README (masks / replay video) |
| **CelebA-Spoof** | 625.537 ảnh, 10.177 chủ thể, 43 thuộc tính + 3 loại thiết bị quay | 🟢 [mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Spoof.html), [GitHub](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | Quá lớn để chỉ đánh giá; phù hợp nếu bạn muốn **fine-tune lại** MiniFASNet (đó sẽ là đóng góp mạnh hơn) |
| **CASIA-SURF / CeFA** | 1000 chủ thể, 21.000 clip, 3 modalities RGB + Depth + IR, có chú thích cross-ethnicity | 🟡 [hướng dẫn tải (CVPR challenge)](https://sites.google.com/view/face-anti-spoofing-challenge/dataset-download/casia-surf-cefacvpr2020) | Chỉ hữu ích nếu bạn mở rộng sang RGB-D; nếu không, dùng để so sánh trong related work |
| **3DMAD** | 76.500 frame, 17 người, mask 3D in + depth camera | 🟢 [data.niaid.nih.gov/resources?id=zenodo_4068477](https://data.niaid.nih.gov/resources?id=zenodo_4068477) | Đo trực tiếp giới hạn "không phát hiện mask 3D" của hệ thống |

**Lưu ý kỹ thuật khi chạy đánh giá cross-dataset** (rất quan trọng, nếu không kết
quả sẽ vô nghĩa):
- Các dataset này là **video** → cần script tách frame và chạy đúng pipeline ảnh
  tĩnh của bạn (detect face bằng UltraFace → crop scale 2.7 → resize 80×80).
- Phải giữ đúng tiền xử lý gốc: crop đưa vào MiniFASNet ở **BGR [0, 255]**, không
  chia 255 — đây là pitfall đã được ghi trong README.
- Khi crop/mask theo nhãn, loại các khung hình có mặt quá nhỏ hoặc có >1 mặt.

---

## 2. So khớp khuôn mặt (ArcFace) — FAR / FRR / EER

Module: [face_service.py](ai/app/services/face_service.py), ngưỡng
`AI_MATCH_SCORE=0.30`, `AI_PARTIAL_SCORE=0.20`.

| Dataset | Nội dung | Mức truy cập |
| --- | --- | --- |
| **LFW** | 13.233 ảnh / 5.749 id, 6000 cặp | 🟢 qua InsightFace datasets |
| **AgeDB-30** | 570 id / 12.240 ảnh / 6K cặp, chênh lệch tuổi ≥30 | 🟢 như trên |
| **CFP-FP** | 500 cặp front-front + 1000 cặp front-profile | 🟢 như trên |
| **CALFW** | Cặp cùng cha khác mẹ / song sinh → khó hơn LFW | 🟢 như trên |
| **IJB-C** | ~3500 chủ thể, 95K ảnh, protocol TAR@FAR=1e-4…1e-2 | 🟡 xin NIST/IARPA |

Tất cả LFW / AgeDB-30 / CFP-FP / CALFW đều có pairlist chuẩn + script tải trong
một chỗ:
[insightface/recognition/_datasets_/README.md](https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md).

**Điểm cần nói rõ trong luận văn**: các benchmark này đánh giá *face ↔ face cùng
chất lượng*. Ca khó của bạn là **mặt in nhỏ trên thẻ CCCD ↔ selfie webcam**
(low-res, glare, in màu) → nên bắt buộc phải có thêm một bộ dữ liệu tự thu
(§6) và báo cáo FAR/FRR trên cả hai: public + domain thực tế.

---

## 3. OCR / bố cục giấy tờ tùy thân

Module: [ocr_service.py](ai/app/services/ocr_service.py),
[qr_service.py](ai/app/services/qr_service.py).

| Dataset | Nội dung | Mức truy cập |
| --- | --- | --- |
| **MIDV-2020** | 1000 thẻ giả lập (10 quốc gia: BD, CN-Guangxi, IN, MY, MA, NG, PH, RW, ZA, UK), 1000 clip + 2000 ảnh scan + 1000 ảnh chụp, nhãn MRZ / keypoints / layout | 🟢 [l3i-share.univ-lr.fr/MIDV2020](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html), bài [arXiv:2107.00396](https://arxiv.org/abs/2107.00396) |
| **MIDV-500** | 500 clip, 50 loại giấy tờ (17 ID card, 14 hộ chiếu, 13 bằng lái) | 🟢 [github.com/fcakyon/midv500](https://github.com/fcakyon/midv500) (có helper tải & convert) |
| **NIST SD10** (hộ chiếu / ID đa quốc gia) | Trước đây là chuẩn đánh giá OCR giấy tờ | 🔴 **đã bị NIST rút** — xem [nist.gov/srd/nist-special-database-10](https://www.nist.gov/srd/nist-special-database-10) |

MIDV-2020/500 **không có CCCD Việt Nam**, nhưng dùng được hợp lệ để đánh giá: tỉ
lệ phát hiện góc thẻ, độ bền OCR với ảnh chụp bằng điện thoại, so sánh với các
nghiên cứu khác. Đây là lý do bạn vẫn cần bộ dữ liệu CCCD tự thu (§6).

---

## 4. Dữ Liệu CCCD Việt Nam

| Nguồn | Nội dung | Mức truy cập / cảnh báo |
| --- | --- | --- |
| **Vietnamese ID Card (Roboflow Universe)** | ~1500 ảnh thẻ, nhãn đối tượng theo trường: `id`, `name`, `dob`, `gender`, `native place`, `expire_date`… (CC BY 4.0) | 🟢 [universe.roboflow.com/.../vietnamese-id-card-uuil0](https://universe.roboflow.com/vehicle-xxgod/vietnamese-id-card-uuil0/dataset/1) |
| **dob-cccd (Roboflow)** | Dataset tập trung trường ngày sinh trên CCCD | 🟢 [universe.roboflow.com/vietnamese-id-card/dob-cccd/dataset/3](https://universe.roboflow.com/vietnamese-id-card/dob-cccd/dataset/3) |
| **New Vietnamese ID Card Dataset (Kaggle)** | Mirror từ Roboflow | 🟢 [kaggle.com/datasets/haidangg/new-vietnamese-id-card-dataset](https://www.kaggle.com/datasets/haidangg/new-vietnamese-id-card-dataset) |
| **phobert-cccd-ner (HuggingFace)** | Mô hình NER 13 nhãn cho trường CCCD — dùng làm baseline so sánh, không phải dataset ảnh | 🟢 [huggingface.co/ngocthanhdoan/phobert-cccd-ner](https://huggingface.co/ngocthanhdoan/phobert-cccd-ner) |

**Cảnh báo bắt buộc trước khi dùng**: các dataset cộng đồng này có thể chứa ảnh
CCCD **thật** (PII của người khác). Với luận văn, bạn phải: (a) kiểm tra nguồn,
(b) ưu tiên thẻ giả lập / đã làm mờ, (c) ghi rõ trong phần đạo đức dữ liệu. Nếu
không chắc, dùng chúng chỉ để **so sánh method**, còn số liệu chính lấy từ bộ tự
thu có sự đồng thuận (§6).

---

## 5. Phát hiện chỉnh sửa thẻ (TamperNet)

Module: [tamper_service.py](ai/app/services/tamper_service.py),
[tamper_net.py](ai/app/services/tamper_net.py).

| Dataset | Nội dung | Mức truy cập |
| --- | --- | --- |
| **FantasyID (Idiap, IJCB 2025)** ⭐ | 13 template thẻ giả tưởng, 13 ngôn ngữ; `bonafide` + `attack/digital_1` + `digital_2` (face swap, inpaint text), **có JSON regions với `region_provenance == "altered"`** → vừa đánh giá binary vừa đánh giá localization | 🟢 [zenodo.org/records/17063366](https://zenodo.org/records/17063366) (2.6 GB, open license; phần giấy phép khác: [17063494](https://zenodo.org/records/17063494)) |
| **DocTamper (CVPR 2023)** | Dataset lớn ảnh tài liệu bị sửa chữ, tổng hợp bán tự động, kèm mask | 🟢 [github.com/qcf-568/DocTamper](https://github.com/qcf-568/DocTamper) |
| **SIDTD** | ID giả mạo sinh tổng hợp dựa trên MIDV-2020 | 🟢 [github.com/Oriolrt/SIDTD_Dataset](https://github.com/Oriolrt/SIDTD_Dataset) |
| **IDNet (2024)** | ID documents sinh bằng few-shot generation | 🟢 [bài báo (NSF PAR)](https://par.nsf.gov/servlets/purl/10617328) |

⭐ **FantasyID là lựa chọn ưu tiên số 1** vì repo của bạn **đã có sẵn công cụ**
cho nó: [prepare_fantasyid.py](ai/training/prepare_fantasyid.py) chuyển FantasyID
sang layout `bona_fide/ forged/ masks/`, và `models/fantasyid_val.txt` là split
val đã dùng. Nghĩa là bạn đã có một tập đánh giá tamper **công khai, reproducible,
có trích dẫn học thuật** — chỉ cần viết script đo (accuracy, IoU vùng bị sửa) là
có ngay số liệu cho Chương 4.

---

## 6. Dữ liệu tự thu (không có bản công khai thay thế)

Những phần sau **không tồn tại dataset public nào** khớp trực tiếp với hệ thống
của bạn → phải tự xây, và chính phần này tạo nên tính "nghiên cứu" của luận văn:

| Cần cho | Cách xây | Quy mô đề xuất |
| --- | --- | --- |
| OCR trường CCCD Việt Nam | Chụp thẻ (xin tình nguyện viên / thẻ mẫu), gán nhãn tay bằng LabelMe hoặc CVAT; sinh thêm biến thể mờ/nghiêng/sáng | 50–100 thẻ |
| QR mặt sau CCCD | Dùng [make_qr_fixture.py](ai/tests/make_qr_fixture.py) sinh QR theo đúng format `|` của bạn + quét vài thẻ thật để kiểm tra format | 30–50 thẻ thật + N QR giả lập |
| Face match thẻ ↔ selfie | Cùng cặp ảnh từ bộ OCR ở trên (mặt in trên thẻ + selfie webcam của cùng/khác người) | ≥30 cặp cùng người, ≥30 cặp khác người |
| In giấy / replay màn hình (webcam thực tế) | Chụp lại ảnh in và màn hình điện thoại của cùng chủ thể, bằng đúng webcam dùng trong app | ≥20 chủ thể × 3 dạng |
| Tấn công "frame tĩnh" cho active liveness | Đưa ảnh tĩnh trước camera, không quay đầu; clip không làm theo yêu cầu | ≥20 clip |

---

## 7. Liveness chủ động / head pose

Module: [active_liveness_service.py](ai/app/services/active_liveness_service.py).

| Dataset | Nội dung | Mức truy cập |
| --- | --- | --- |
| **AFLW2000-3D** | 2000 ảnh, nhãn pose 6 DoF (yaw/pitch/roll) — test set chuẩn | 🟢 (trang AFLW, UCSD CVRC) |
| **300W-LP** | 122.450 ảnh render 3DMM, hiệu chuẩn pose — train set chuẩn | 🟢 (proj.3dfacealign) |
| **BIWI / FHRI** | Video head pose chuẩn xác, nhiều hướng | 🟡 xin Univ. of Tübingen |

Giao thức chuẩn trong tài liệu: train trên 300W-LP → test trên AFLW2000-3D (xem
[bài tổng quan](https://arxiv.org/html/2407.05357v3),
[repo tham chiếu](https://github.com/yakhyo/head-pose-estimation)).

Cách dùng cho luận văn: bạn **không** train lại mô hình, mà dùng các ảnh + nhãn
yaw này để **kiểm định độ chính xác của MediaPipe FaceLandmarker** (MAE theo yaw)
trước khi tin vào ngưỡng "quay trái/quay phải" — biến một bước kiểm thử thủ công
thành một experiment có số.

---

## 8. Bộ dataset tối thiểu nên chọn (lộ trình nhanh)

Nếu thời gian hạn chế, 4 nguồn sau cho bạn ~80% số liệu cần thiết với chi phí thấp:

1. **Replay-Attack (🟢)** + **MSU-MFSD (🟡)** → APCER/BPCER cho passive liveness.
2. **FantasyID (🟢)** → accuracy + localization cho tamper (hạ tầng đã có sẵn).
3. **LFW + AgeDB-30 + CFP-FP qua InsightFace (🟢)** → đường cong ROC/EER cho
   ArcFace; chứng minh ngưỡng 0.30 là hợp lý.
4. **Bộ tự thu CCCD + in giấy (🔧)** → số liệu *trong miền thực tế* mà reviewer
   sẽ hỏi đầu tiên.

Cộng thêm nếu còn thời gian: **OULU-NPU** (biến đổi thiết bị/ánh sáng),
**ROSE-Youtu** (mask giấy 3D), **MIDV-2020** (độ bền OCR theo chất lượng chụp).

---

## 9. Ghi chú về trích dẫn & giấy phép

- Khi dùng dataset, trích **bài báo gốc** (không chỉ link tải). Ví dụ: FantasyID —
  Korshunov, Mohammadi, Vidit, Ecabert, Marcel, *FantasyID: A dataset for detecting
  digital manipulations of ID-documents*, IJCB 2025.
- 🟡 "thỏa thuận" thường cấm dùng cho mục đích thương mại → ghi rõ trong luận văn
  là dùng cho nghiên cứu học thuật.
- Luôn ghi **phiên bản dataset + ngày tải + checksum** để bảo vệ tính
  reproducible (Hội đồng ngày càng hỏi điểm này).
- Với dữ liệu tự thu: kèm **form đồng thuận** và mô tả **quy trình ẩn danh / hủy
  dữ liệu** trong phụ lục — đây là điểm cộng lớn cho đề tài có dữ liệu sinh trắc.
