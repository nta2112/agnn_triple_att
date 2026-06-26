import streamlit as st
import torch
import torch.nn as nn
import os
import sys

# Tránh lỗi sập tiến trình đột ngột (OMP: Error #15) do xung đột thư viện OpenMP trên Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import shutil
import importlib.util
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Giải quyết lỗi tương thích ngược giữa NumPy 2.x và NumPy 1.x khi load checkpoint bằng PyTorch
try:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
except ImportError:
    pass

import pandas as pd

# Import các module từ codebase hiện tại
from backbone import ResNet12, ConvNet, ResNet50Pretrained
from agnn import AGNN
from utils import allocate_tensors, initialize_nodes_edges, backbone_two_stage_initialization, one_hot_encode, label2edge

# ── Cấu hình trang Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title="AGNN Few-Shot Demo",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách Glassmorphism và Dark Mode cao cấp
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Thiết lập font chữ chính */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Kiểu dáng cho các box thông tin */
    .card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(108, 92, 231, 0.1), rgba(0, 206, 201, 0.1));
        border-radius: 12px;
        border: 1px solid rgba(108, 92, 231, 0.2);
        padding: 16px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #00cec9;
    }
    
    .metric-label {
        font-size: 14px;
        color: #a29bfe;
    }
    
    /* Hiệu ứng cho nút bấm */
    .stButton>button {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white !important;
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.5);
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
    }
    
    /* Custom style cho các tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        color: #858585;
        padding: 12px 16px;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #00cec9;
    }

    .stTabs [aria-selected="true"] {
        color: #00cec9 !important;
        border-bottom-color: #00cec9 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Khai báo đường dẫn mặc định ───────────────────────────────────────
DEFAULT_SUPPORT_DIR = os.path.join("app", "demo_support")
CONFIG_PATH = os.path.join("app", "config.py")
MODEL_PATH = os.path.join("app", "model_best.pth.tar")
LOGO_PATH = os.path.join("app", "logo_tlu.png")

# ── Hàm sinh dữ liệu mẫu ngẫu nhiên (Mock Support Set) ────────────────
def generate_mock_data(support_dir):
    """Tự động tạo 5 lớp quả với 3 ảnh màu trơn + nhiễu mỗi lớp để demo có thể chạy ngay lập tức"""
    os.makedirs(support_dir, exist_ok=True)
    classes = {
        "Tao_Do": (235, 77, 75),       # Đỏ
        "Tao_Xanh": (106, 176, 76),    # Xanh lá
        "Cam_Cam": (240, 147, 43),     # Cam
        "Nho_Tim": (190, 46, 221),     # Tím
        "Chuoi_Vang": (241, 196, 15)   # Vàng
    }
    
    for cname, color in classes.items():
        cpath = os.path.join(support_dir, cname)
        if not os.path.exists(cpath):
            os.makedirs(cpath)
            for i in range(3):
                # Tạo ảnh màu cơ bản
                img = Image.new("RGB", (128, 128), color=color)
                draw = ImageDraw.Draw(img)
                # Vẽ một vài hình tròn ngẫu nhiên để tạo chút cấu trúc/nhiễu
                for _ in range(5):
                    x0 = np.random.randint(0, 80)
                    y0 = np.random.randint(0, 80)
                    r = np.random.randint(10, 40)
                    draw.ellipse([x0, y0, x0+r, y0+r], fill=(
                        max(0, min(255, color[0] + np.random.randint(-30, 30))),
                        max(0, min(255, color[1] + np.random.randint(-30, 30))),
                        max(0, min(255, color[2] + np.random.randint(-30, 30)))
                    ))
                img.save(os.path.join(cpath, f"sample_{i+1}.jpg"))

def list_classes(support_dir):
    if not os.path.exists(support_dir):
        return []
    return sorted([d for d in os.listdir(support_dir) if os.path.isdir(os.path.join(support_dir, d))])


def safe_filename(name):
    stem = Path(name).stem.replace(" ", "_")
    suffix = Path(name).suffix.lower()
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(keep).strip("_")[:80] + suffix


def load_checkpoint_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location=map_location)

# ── Tiền xử lý ảnh (Image Transforms) ─────────────────────────────────
def get_transform(image_size):
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    box_size = int(image_size * 1.15) if image_size > 0 else 96

    def transform(img):
        img = img.convert("RGB")
        bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
        img = img.resize((box_size, box_size), resample=bicubic)

        left = max(0, (box_size - image_size) // 2)
        top = max(0, (box_size - image_size) // 2)
        img = img.crop((left, top, left + image_size, top + image_size))

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor(mean_pix, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(std_pix, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    return transform

# ── Hàm nạp Mô hình (Cached để tránh nạp lại nhiều lần) ────────────────
@st.cache_resource
def load_models(config_path, checkpoint_path, device):
    # 1. Đọc file cấu hình
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
    
    # 2. Khởi tạo Backbone
    if config['backbone'] == 'resnet12':
        enc_module = ResNet12(emb_size=config['emb_size'])
    elif config['backbone'] == 'resnet50':
        enc_module = ResNet50Pretrained(emb_size=config['emb_size'])
    elif config['backbone'] == 'convnet':
        enc_module = ConvNet(emb_size=config['emb_size'])
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']}")
        
    # 3. Nạp Trọng số
    checkpoint = load_checkpoint_compat(checkpoint_path, map_location=device)
    
    def clean_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict
        
    enc_module.load_state_dict(clean_state_dict(checkpoint['enc_module_state_dict']))
    enc_module.to(device).eval()
    
    # Lưu lại checkpoint của GNN để khởi tạo động theo kích thước Support Set sau này
    gnn_state_dict = clean_state_dict(checkpoint['gnn_module_state_dict'])
    
    return enc_module, gnn_state_dict, config

# ── Sidebar Cấu hình ──────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ HỆ THỐNG AGNN</h2>", unsafe_allow_html=True)
device_choice = st.sidebar.selectbox("Thiết bị xử lý:", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
device = torch.device(device_choice)

support_dir_input = st.sidebar.text_input(
    "Thư mục Support Set:",
    value=st.session_state.get("support_dir", DEFAULT_SUPPORT_DIR),
    help="Mỗi lớp là một thư mục con, ví dụ support/Tao_Do/*.jpg"
)
SUPPORT_DIR = support_dir_input.strip() or DEFAULT_SUPPORT_DIR
st.session_state["support_dir"] = SUPPORT_DIR
os.makedirs(SUPPORT_DIR, exist_ok=True)

if st.sidebar.checkbox("Tạo dữ liệu mẫu nếu thiếu lớp", value=False):
    if len(list_classes(SUPPORT_DIR)) < 5:
        generate_mock_data(SUPPORT_DIR)
        st.sidebar.success("Đã tạo support set mẫu.")
        st.rerun()

# Nạp model
with st.spinner("Đang tải mô hình lên RAM..."):
    enc_module = None
    gnn_state_dict = None
    config = None
    try:
        enc_module, gnn_state_dict, config = load_models(CONFIG_PATH, MODEL_PATH, device)
        st.sidebar.success("✓ Đã nạp thành công mô hình!")
    except Exception as e:
        st.sidebar.error(f"Lỗi nạp mô hình: {e}")
        st.stop()

if config is None:
    st.stop()

# Hiển thị thông số cấu hình của Model trong Sidebar
st.sidebar.markdown("### 📊 Thông tin Mô hình")
st.sidebar.info(f"""
- **Backbone:** {config['backbone'].upper()}
- **Kích thước Embedding:** {config['emb_size']}
- **Số lớp GNN (Generations):** {config['num_generation']}
- **Hàm đo khoảng cách:** {config['point_distance_metric'].upper()}
""")

# ── Tích hợp lớp đã học (Base Classes) ──
use_base_classes = st.sidebar.checkbox(
    "Tích hợp lớp đã học (Base Classes)", 
    value=False,
    help="Sử dụng Nút ảo Prototype của các lớp đã huấn luyện mà không cần nạp ảnh."
)

PROTOTH_PATH = os.path.join("app", "base_prototypes.pth")
base_classes = []
base_proto_last = None
base_proto_second = None

if use_base_classes:
    if os.path.exists(PROTOTH_PATH):
        try:
            base_data = torch.load(PROTOTH_PATH, map_location=device, weights_only=False)
            base_classes = base_data['class_names']
            base_proto_last = base_data['prototypes_last'].to(device)
            base_proto_second = base_data['prototypes_second'].to(device)
            st.sidebar.success(f"✓ Đã nạp {len(base_classes)} lớp đã học!")
        except Exception as e:
            st.sidebar.error(f"Lỗi nạp file prototypes: {e}")
    else:
        st.sidebar.warning("⚠ Chưa tìm thấy base_prototypes.pth")
        if st.sidebar.button("⚙️ Khởi tạo Ký ức Lớp đã học"):
            with st.spinner("Đang tính toán vector Prototype cho các lớp đã học..."):
                try:
                    from compute_prototypes import SimpleDataset
                    from torch.utils.data import DataLoader as TorchDataLoader
                    
                    dataset_p = SimpleDataset(root=SUPPORT_DIR, image_size=config['image_size'])
                    data_loader_p = TorchDataLoader(dataset_p, batch_size=64, shuffle=False)
                    
                    class_features_last = [[] for _ in range(len(dataset_p.class_names))]
                    class_features_second = [[] for _ in range(len(dataset_p.class_names))]
                    
                    with torch.no_grad():
                        for imgs, labels in data_loader_p:
                            imgs = imgs.to(device)
                            imgs_expanded = imgs.unsqueeze(1)
                            last_feat, second_last_feat = backbone_two_stage_initialization(imgs_expanded, enc_module)
                            
                            last_feat = last_feat.squeeze(1).cpu()
                            second_last_feat = second_last_feat.squeeze(1).cpu()
                            
                            for i, label in enumerate(labels):
                                class_idx = label.item()
                                class_features_last[class_idx].append(last_feat[i])
                                class_features_second[class_idx].append(second_last_feat[i])
                                
                    prototypes_last = []
                    prototypes_second = []
                    for class_idx in range(len(dataset_p.class_names)):
                        mean_last = torch.mean(torch.stack(class_features_last[class_idx]), dim=0)
                        mean_second = torch.mean(torch.stack(class_features_second[class_idx]), dim=0)
                        prototypes_last.append(mean_last)
                        prototypes_second.append(mean_second)
                        
                    torch.save({
                        'class_names': dataset_p.class_names,
                        'prototypes_last': torch.stack(prototypes_last),
                        'prototypes_second': torch.stack(prototypes_second),
                        'emb_size': config['emb_size']
                    }, PROTOTH_PATH)
                    
                    st.sidebar.success("✓ Khởi tạo Ký ức thành công!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Lỗi khởi tạo: {e}")

if st.sidebar.button("🧹 Xóa cache & Tải lại"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# ── Giao diện Chính ───────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    logo_col, title_col, spacer_col = st.columns([1, 4, 1])
    with logo_col:
        st.image(LOGO_PATH, use_container_width=True)
    with title_col:
        st.markdown("<h1 style='text-align: center; color: #00cec9;'>🍇 AGNN FEW-SHOT DEMO</h1>", unsafe_allow_html=True)
    with spacer_col:
        st.empty()
else:
    st.markdown("<h1 style='text-align: center; color: #00cec9;'>🍇 AGNN FEW-SHOT DEMO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #a29bfe;'>Phân loại ảnh Nông sản thích ứng nhanh bằng Đồ thị thông tin (Graph Neural Networks)</p>", unsafe_allow_html=True)

# Lấy danh sách các lớp hiện có trong thư mục support
all_dirs = list_classes(SUPPORT_DIR)

# Thiết lập Tab
tab_predict, tab_manage = st.tabs(["🚀 Dự đoán (Inference)", "📁 Quản lý Dữ liệu Mẫu (Support Set)"])

# ──────────────────────────────────────────────────────────────────────
# TAB 1: DỰ ĐOÁN & PHÂN TÍCH ĐỒ THỊ
# ──────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    if not use_base_classes:
        # ── Chế độ Few-Shot thuần túy (Original Logic) ──
        st.subheader("1. Lập Tác vụ Phân loại (5-Way Task Setup)")
        st.caption(f"Support set đang dùng: {SUPPORT_DIR}")
        
        if len(all_dirs) < 5:
            st.warning(f"Hiện tại chỉ có {len(all_dirs)} lớp trong thư mục Support Set. Vui lòng thêm thêm lớp ở Tab 'Quản lý Dữ liệu Mẫu' để đủ ít nhất 5 lớp cho mô hình 5-Way hoạt động.")
        else:
            selected_classes = st.multiselect(
                "Chọn đúng 5 lớp quả để đưa vào đồ thị đối chiếu:",
                options=all_dirs,
                default=all_dirs[:5],
                max_selections=5
            )
            
            if len(selected_classes) == 5:
                class_counts = []
                for cname in selected_classes:
                    cpath = os.path.join(SUPPORT_DIR, cname)
                    imgs = [f for f in os.listdir(cpath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    class_counts.append(len(imgs))
                min_shots = min(class_counts)
                
                if min_shots == 0:
                    st.error("Một số lớp được chọn đang không có ảnh mẫu nào. Hãy thêm ảnh mẫu trước.")
                else:
                    st.markdown(f"**Số lượng ảnh mẫu tìm thấy tối thiểu:** `{min_shots}` ảnh mỗi lớp.")
                    shot_choice = st.slider(
                        "Chọn số ảnh mẫu đối chiếu mỗi lớp (K-Shot):",
                        min_value=1,
                        max_value=min_shots,
                        value=min_shots if min_shots < 5 else 5
                    )
                    
                    st.markdown("---")
                    st.subheader("2. Tải ảnh quả cần dự đoán lên (Query Image)")
                    query_file = st.file_uploader("Kéo thả ảnh cần dự đoán vào đây...", type=["jpg", "png", "jpeg", "webp"])
                    
                    if query_file is not None:
                        col_q, col_p = st.columns([1, 1])
                        with col_q:
                            query_image = Image.open(query_file).convert('RGB')
                            st.image(query_image, caption="Ảnh Query cần nhận diện", use_container_width=True)
                        with col_p:
                            st.markdown("<br><br>", unsafe_allow_html=True)
                            predict_btn = st.button("🔍 Phân tích mạng Đồ thị & Dự đoán")
                            
                        if predict_btn:
                            with st.spinner("Đang chạy lan truyền đồ thị (AGNN Forward)..."):
                                try:
                                    transform = get_transform(config['image_size'])
                                    all_support_data = []
                                    all_support_labels = []
                                    support_image_paths = []
                                    
                                    for idx, cname in enumerate(selected_classes):
                                        cpath = os.path.join(SUPPORT_DIR, cname)
                                        img_names = sorted([f for f in os.listdir(cpath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])[:shot_choice]
                                        for fname in img_names:
                                            fpath = os.path.join(cpath, fname)
                                            img = Image.open(fpath).convert('RGB')
                                            all_support_data.append(transform(img))
                                            all_support_labels.append(idx)
                                            support_image_paths.append((cname, fpath))
                                            
                                    support_data = torch.stack(all_support_data).unsqueeze(0)
                                    support_label = torch.tensor(all_support_labels).unsqueeze(0)
                                    num_total_supports = support_data.size(1)
                                    
                                    query_data = transform(query_image).unsqueeze(0).unsqueeze(0)
                                    query_label = torch.zeros((1, 1), dtype=torch.long)
                                    
                                    tensors = allocate_tensors()
                                    batch = (support_data.unsqueeze(0), 
                                             support_label.unsqueeze(0), 
                                             query_data.unsqueeze(0), 
                                             query_label.unsqueeze(0))
                                    
                                    _, support_label_node, _, _, all_data, _, node_feature_gd, edge_feature_gp = \
                                        initialize_nodes_edges(batch, num_total_supports, tensors, 1, 1, 1, device)
                                    
                                    all_data = all_data.to(device)
                                    node_feature_gd = node_feature_gd.to(device)
                                    edge_feature_gp = edge_feature_gp.to(device)
                                    support_label_node = support_label_node.to(device)
                                    
                                    gnn_module = AGNN(in_c=config['emb_size'],
                                                      num_generations=config['num_generation'],
                                                      dropout=config['train_config']['dropout'],
                                                      num_support_sample=num_total_supports,
                                                      num_sample=num_total_supports + 1,
                                                      loss_indicator=config['train_config']['loss_indicator'],
                                                      point_metric=config['point_distance_metric'],
                                                      ablation_mode=config.get('ablation_mode', 'full'))
                                    gnn_module.load_state_dict(gnn_state_dict)
                                    gnn_module.to(device).eval()
                                    
                                    with torch.no_grad():
                                        last_layer_data, second_last_layer_data = backbone_two_stage_initialization(all_data, enc_module)
                                        point_similarities, _, _ = gnn_module(second_last_layer_data,
                                                                              last_layer_data,
                                                                              node_feature_gd,
                                                                              edge_feature_gp,
                                                                              support_label_node)
                                                                              
                                    last_similarity = point_similarities[-1]
                                    query_sim = last_similarity[:, num_total_supports:, :num_total_supports]
                                    one_hot_support = one_hot_encode(5, support_label_node.long(), device)
                                    query_node_pred = torch.bmm(query_sim, one_hot_support)
                                    
                                    probabilities = torch.softmax(query_node_pred, dim=-1).squeeze().cpu().numpy()
                                    pred_idx = np.argmax(probabilities)
                                    confidence = probabilities[pred_idx]
                                    pred_class = selected_classes[pred_idx]
                                    
                                    # Trực quan
                                    st.markdown("---")
                                    st.subheader("🎯 KẾT QUẢ PHÂN LOẠI")
                                    
                                    col_res1, col_res2 = st.columns([1, 1])
                                    with col_res1:
                                        st.markdown(f"""
                                        <div class='metric-card'>
                                            <div class='metric-label'>Lớp quả dự đoán</div>
                                            <div class='metric-value'>{pred_class}</div>
                                            <div class='metric-label'>Độ tin cậy: {confidence:.2%}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        chart_data = pd.DataFrame({
                                            'Lớp': selected_classes,
                                            'Xác suất (%)': probabilities * 100
                                        }).set_index('Lớp')
                                        st.bar_chart(chart_data)
                                        
                                    with col_res2:
                                        st.markdown("#### 🔗 Phân tích Liên kết Đồ thị (Graph Attention Link)")
                                        st.write("Mô hình AGNN dựa trên liên kết mạnh nhất giữa nút Query và các nút Support để dự đoán:")
                                        support_weights = query_sim.squeeze(0).squeeze(0).cpu().numpy()
                                        best_sup_idx = np.argmax(support_weights)
                                        best_sup_weight = support_weights[best_sup_idx]
                                        best_sup_class, best_sup_img_path = support_image_paths[best_sup_idx]
                                        
                                        best_sup_img = Image.open(best_sup_img_path)
                                        st.image(best_sup_img, caption=f"Ảnh mẫu giống nhất: {best_sup_class} (Trọng số: {best_sup_weight:.4f})", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Lỗi dự đoán: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
            else:
                st.info("Vui lòng chọn chính xác 5 lớp quả để tiếp tục.")

    else:
        # ── Chế độ Hybrid GFSL (Base Classes + Novel Classes) ──
        st.subheader("1. Lập Tác vụ Phân loại lai (Hybrid GFSL Setup)")
        st.caption(f"Đang sử dụng các lớp đã học từ file Ký ức làm Nút ảo.")
        
        if len(base_classes) == 0:
            st.error("Chưa nạp được các lớp đã học. Vui lòng kiểm tra hoặc khởi tạo file Ký ức ở Sidebar.")
        else:
            st.write(f"**Các lớp đã học tích hợp sẵn:** `{', '.join(base_classes)}`")
            
            # Cho phép chọn thêm lớp mới nếu có
            selected_novel_classes = st.multiselect(
                "Chọn thêm các lớp mới (Novel Classes) nếu có:",
                options=[d for d in all_dirs if d not in base_classes],
                default=[]
            )
            
            shot_choice = 0
            has_novel_data = True
            if len(selected_novel_classes) > 0:
                class_counts = []
                for cname in selected_novel_classes:
                    cpath = os.path.join(SUPPORT_DIR, cname)
                    imgs = [f for f in os.listdir(cpath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    class_counts.append(len(imgs))
                min_shots = min(class_counts)
                
                if min_shots == 0:
                    st.error("Một số lớp mới được chọn không có ảnh mẫu nào. Hãy thêm ảnh mẫu trước.")
                    has_novel_data = False
                else:
                    st.markdown(f"**Số lượng ảnh mẫu lớp mới tối thiểu:** `{min_shots}` ảnh mỗi lớp.")
                    shot_choice = st.slider(
                        "Chọn số ảnh mẫu lớp mới đối chiếu (K-Shot):",
                        min_value=1,
                        max_value=min_shots,
                        value=min_shots if min_shots < 5 else 5
                    )
            
            if has_novel_data:
                st.markdown("---")
                st.subheader("2. Tải ảnh quả cần dự đoán lên (Query Image)")
                query_file = st.file_uploader("Kéo thả ảnh cần dự đoán vào đây...", type=["jpg", "png", "jpeg", "webp"])
                
                if query_file is not None:
                    col_q, col_p = st.columns([1, 1])
                    with col_q:
                        query_image = Image.open(query_file).convert('RGB')
                        st.image(query_image, caption="Ảnh Query cần nhận diện", use_container_width=True)
                    with col_p:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        predict_btn = st.button("🔍 Phân tích mạng Đồ thị & Dự đoán")
                        
                    if predict_btn:
                        with st.spinner("Đang chạy lan truyền đồ thị Hybrid..."):
                            try:
                                transform = get_transform(config['image_size'])
                                
                                # Trích xuất đặc trưng Query
                                query_data = transform(query_image).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    query_last, query_second = backbone_two_stage_initialization(query_data.unsqueeze(0), enc_module)
                                
                                # Chuẩn bị Novel support features
                                novel_features_last = []
                                novel_features_second = []
                                novel_labels = []
                                novel_image_paths = []
                                
                                num_base = len(base_classes)
                                for idx, cname in enumerate(selected_novel_classes):
                                    cpath = os.path.join(SUPPORT_DIR, cname)
                                    img_names = sorted([f for f in os.listdir(cpath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])[:shot_choice]
                                    
                                    novel_imgs = []
                                    for fname in img_names:
                                        fpath = os.path.join(cpath, fname)
                                        img = Image.open(fpath).convert('RGB')
                                        novel_imgs.append(transform(img))
                                        novel_image_paths.append((cname, fpath))
                                        
                                    novel_imgs_tensor = torch.stack(novel_imgs).to(device)
                                    with torch.no_grad():
                                        last, second = backbone_two_stage_initialization(novel_imgs_tensor.unsqueeze(0), enc_module)
                                    novel_features_last.append(last.squeeze(0))
                                    novel_features_second.append(second.squeeze(0))
                                    novel_labels.append(torch.full((len(novel_imgs),), num_base + idx, dtype=torch.long, device=device))
                                
                                # Kết hợp Base prototypes (Nút ảo) và Novel features (Nút thực)
                                combined_last = [base_proto_last]
                                combined_second = [base_proto_second]
                                combined_labels = [torch.arange(num_base, dtype=torch.long, device=device)]
                                
                                if len(selected_novel_classes) > 0:
                                    combined_last.append(torch.cat(novel_features_last, dim=0))
                                    combined_second.append(torch.cat(novel_features_second, dim=0))
                                    combined_labels.append(torch.cat(novel_labels, dim=0))
                                    
                                support_features_last = torch.cat(combined_last, dim=0)
                                support_features_second = torch.cat(combined_second, dim=0)
                                support_label = torch.cat(combined_labels, dim=0)
                                num_total_supports = support_features_last.size(0)
                                
                                # Chuẩn bị tensor đầu vào cho GNN
                                last_layer_data = torch.cat([support_features_last.unsqueeze(0), query_last], dim=1)
                                second_last_layer_data = torch.cat([support_features_second.unsqueeze(0), query_second], dim=1)
                                
                                support_label_exp = support_label.unsqueeze(0)
                                
                                node_gd_init_support = label2edge(support_label_exp, device)
                                node_gd_init_query = (torch.ones([1, 1, num_total_supports]) * torch.tensor(1. / num_total_supports)).to(device)
                                node_feature_gd = torch.cat([node_gd_init_support, node_gd_init_query], dim=1)
                                
                                num_total_nodes = num_total_supports + 1
                                edge_feature_gp = torch.zeros(1, num_total_nodes, num_total_nodes, device=device)
                                edge_feature_gp[:, :num_total_supports, :num_total_supports] = node_gd_init_support
                                edge_feature_gp[:, num_total_supports:, :num_total_supports] = 1. / num_total_supports
                                edge_feature_gp[:, :num_total_supports, num_total_supports:] = 1. / num_total_supports
                                edge_feature_gp[:, num_total_supports, num_total_supports] = 1.0
                                
                                # Khởi tạo AGNN model động
                                gnn_module = AGNN(in_c=config['emb_size'],
                                                  num_generations=config['num_generation'],
                                                  dropout=config['train_config']['dropout'],
                                                  num_support_sample=num_total_supports,
                                                  num_sample=num_total_supports + 1,
                                                  loss_indicator=config['train_config']['loss_indicator'],
                                                  point_metric=config['point_distance_metric'],
                                                  ablation_mode=config.get('ablation_mode', 'full'))
                                gnn_module.load_state_dict(gnn_state_dict)
                                gnn_module.to(device).eval()
                                
                                # Forward Pass GNN
                                with torch.no_grad():
                                    point_similarities, _, _ = gnn_module(second_last_layer_data,
                                                                          last_layer_data,
                                                                          node_feature_gd,
                                                                          edge_feature_gp,
                                                                          support_label_exp)
                                                                          
                                # Lấy dự đoán
                                last_similarity = point_similarities[-1]
                                query_sim = last_similarity[:, num_total_supports:, :num_total_supports]
                                
                                class_names = base_classes + selected_novel_classes
                                num_ways = len(class_names)
                                one_hot_support = one_hot_encode(num_ways, support_label_exp.long(), device)
                                query_node_pred = torch.bmm(query_sim, one_hot_support)
                                
                                probabilities = torch.softmax(query_node_pred, dim=-1).squeeze().cpu().numpy()
                                pred_idx = np.argmax(probabilities)
                                confidence = probabilities[pred_idx]
                                pred_class = class_names[pred_idx]
                                
                                # Trực quan hóa kết quả
                                st.markdown("---")
                                st.subheader("🎯 KẾT QUẢ PHÂN LOẠI")
                                
                                col_res1, col_res2 = st.columns([1, 1])
                                with col_res1:
                                    st.markdown(f"""
                                    <div class='metric-card'>
                                        <div class='metric-label'>Lớp quả dự đoán</div>
                                        <div class='metric-value'>{pred_class}</div>
                                        <div class='metric-label'>Độ tin cậy: {confidence:.2%}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Hiển thị Top 5 lớp có xác suất cao nhất
                                    top_k = min(5, len(class_names))
                                    top_indices = np.argsort(probabilities)[::-1][:top_k]
                                    top_classes = [class_names[i] for i in top_indices]
                                    top_probs = [probabilities[i] * 100 for i in top_indices]
                                    
                                    chart_data = pd.DataFrame({
                                        'Lớp': top_classes,
                                        'Xác suất (%)': top_probs
                                    }).set_index('Lớp')
                                    st.bar_chart(chart_data)
                                    
                                with col_res2:
                                    st.markdown("#### 🔗 Phân tích Liên kết Đồ thị (Graph Attention Link)")
                                    st.write("Mô hình AGNN dựa trên liên kết mạnh nhất giữa nút Query và các nút Support để dự đoán:")
                                    
                                    support_weights = query_sim.squeeze(0).squeeze(0).cpu().numpy()
                                    best_sup_idx = np.argmax(support_weights)
                                    best_sup_weight = support_weights[best_sup_idx]
                                    
                                    if best_sup_idx < num_base:
                                        st.info(f"Nút có liên kết mạnh nhất là **Nút ảo (Virtual Node)** đại diện cho lớp cũ đã học: **{class_names[best_sup_idx]}** (Trọng số liên kết: {best_sup_weight:.4f})")
                                    else:
                                        novel_idx = best_sup_idx - num_base
                                        best_sup_class, best_sup_img_path = novel_image_paths[novel_idx]
                                        best_sup_img = Image.open(best_sup_img_path)
                                        st.image(best_sup_img, caption=f"Ảnh mẫu lớp mới giống nhất: {best_sup_class} (Trọng số liên kết: {best_sup_weight:.4f})", use_container_width=True)
                            except Exception as e:
                                st.error(f"Lỗi dự đoán: {e}")
                                import traceback
                                st.code(traceback.format_exc())
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# TAB 2: QUẢN LÝ DỮ LIỆU MẪU (SUPPORT SET)
# ──────────────────────────────────────────────────────────────────────
with tab_manage:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📂 Kho Dữ liệu Mẫu (Support Set Database)")
    st.write(f"Bộ nhớ dữ liệu mẫu hiện đang lưu trữ tại: `{SUPPORT_DIR}`")
    
    # Hiển thị số lượng ảnh của từng lớp
    st.markdown("#### Thống kê các lớp hiện có:")
    stats_data = []
    for d in all_dirs:
        cpath = os.path.join(SUPPORT_DIR, d)
        imgs = [f for f in os.listdir(cpath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        stats_data.append({"Lớp quả": d, "Số ảnh mẫu": len(imgs)})
    
    st.table(pd.DataFrame(stats_data))
    
    st.markdown("---")
    
    # Xem ảnh của các lớp
    st.markdown("#### Xem chi tiết ảnh mẫu của các lớp:")
    selected_view_class = st.selectbox("Chọn lớp muốn xem ảnh mẫu:", all_dirs) if all_dirs else None
    if not all_dirs:
        st.warning("Chưa có lớp nào trong Support Set. Hãy tạo lớp mới và tải ảnh lên ở phần bên dưới.")
    elif selected_view_class:
        view_path = os.path.join(SUPPORT_DIR, selected_view_class)
        view_imgs = sorted([f for f in os.listdir(view_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        if len(view_imgs) == 0:
            st.write("Không có ảnh mẫu nào trong lớp này.")
        else:
            # Hiển thị ảnh dạng lưới
            cols = st.columns(5)
            for i, img_name in enumerate(view_imgs):
                col = cols[i % 5]
                img_path = os.path.join(view_path, img_name)
                img = Image.open(img_path)
                col.image(img, caption=img_name, use_container_width=True)
                
                # Nút xóa ảnh
                if col.button("🗑️ Xóa", key=f"del_{selected_view_class}_{img_name}"):
                    os.remove(img_path)
                    st.success(f"Đã xóa {img_name}")
                    st.rerun()
                    
    st.markdown("---")
    
    # Thêm ảnh mẫu mới / Lớp mới
    st.markdown("### ➕ Bổ sung / Thêm lớp quả mới vào mô hình")
    st.info("Ảnh mới được dùng ngay như support images cho các lần dự đoán sau. Nếu muốn cập nhật trọng số mô hình, bạn vẫn cần fine-tune/train lại checkpoint bằng main_gnn.py.")
    col_add1, col_add2 = st.columns([1, 1])
    
    with col_add1:
        add_type = st.radio("Chọn thao tác:", ["Thêm ảnh vào lớp có sẵn", "Tạo lớp quả mới hoàn toàn"])
        
        if add_type == "Thêm ảnh vào lớp có sẵn":
            if all_dirs:
                target_class = st.selectbox("Chọn lớp quả đích:", all_dirs)
            else:
                st.warning("Chưa có lớp có sẵn. Hãy chọn thao tác tạo lớp mới.")
                target_class = ""
        else:
            target_class = st.text_input("Nhập tên lớp quả mới (không dấu, viết liền hoặc dùng gạch dưới, ví dụ: 'Sau_Rieng'):")
            
    with col_add2:
        new_files = st.file_uploader("Tải lên một hoặc nhiều ảnh mẫu cho lớp này:", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True)
        
        if st.button("💾 Lưu vào Support Set"):
            if not target_class or target_class.strip() == "":
                st.error("Vui lòng nhập hoặc chọn tên lớp hợp lệ.")
            elif not new_files:
                st.error("Vui lòng tải lên ít nhất một ảnh.")
            else:
                clean_target_class = target_class.strip().replace(" ", "_")
                class_path = os.path.join(SUPPORT_DIR, clean_target_class)
                os.makedirs(class_path, exist_ok=True)
                
                # Lưu file
                saved_count = 0
                for f in new_files:
                    base_name = safe_filename(f.name)
                    file_dest = os.path.join(class_path, base_name)
                    if os.path.exists(file_dest):
                        stem = Path(base_name).stem
                        suffix = Path(base_name).suffix
                        idx = 1
                        while os.path.exists(file_dest):
                            file_dest = os.path.join(class_path, f"{stem}_{idx}{suffix}")
                            idx += 1
                    with open(file_dest, "wb") as out_f:
                        out_f.write(f.getbuffer())
                    saved_count += 1
                
                st.success(f"✓ Đã lưu thành công {saved_count} ảnh vào lớp '{clean_target_class}'!")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
