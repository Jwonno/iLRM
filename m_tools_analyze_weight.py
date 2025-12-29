import torch
import numpy as np
import matplotlib.pyplot as plt
import io

def analyze_checkpoint(ckpt_path):
    print(f"🔍 Loading checkpoint: {ckpt_path} ...")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model_sd = checkpoint.get('model', {})
    optim_sd = checkpoint.get('optimizer', {})
    param_groups = optim_sd.get('param_groups', [])
    state = optim_sd.get('state', {})

    print(f"✅ Checkpoint Loaded. Step: {checkpoint.get('step', 'Unknown')}")
    
    # 1. Optimizer 설정 확인 (beta2 값 확인)
    if param_groups:
        betas = param_groups[0].get('betas', (0.9, 0.999))
        print(f"ℹ️ Optimizer config found: betas={betas}")
        if betas[1] < 0.9:
            print(f"⚠️ WARNING: Beta2 is unusually low ({betas[1]}). Expect high variance in exp_avg_sq.")
    
    # 분석을 위한 데이터 수집 리스트
    weight_stats = []
    v_t_stats = [] # exp_avg_sq
    update_ratios = [] # |m_t| / sqrt(v_t)
    
    print("\n--- 1. Analyzing Model Weights (Parameters) ---")
    total_params = 0
    nan_params = 0
    large_params = 0 # 절대값이 10.0을 넘는 파라미터 수
    
    for key, param in model_sd.items():
        param_np = param.float().numpy()
        total_params += param_np.size
        
        # NaN / Inf 체크
        if np.isnan(param_np).any() or np.isinf(param_np).any():
            nan_params += 1
            print(f"❌ NaN/Inf detected in layer: {key}")
            
        # 가중치 크기 체크 (발산 징후)
        max_val = np.max(np.abs(param_np))
        if max_val > 10.0: # 보통 가중치는 1.0 미만임
            large_params += 1
            # 상위 3개만 출력
            if large_params <= 3:
                print(f"⚠️ Large weight detected in {key}: max abs value = {max_val:.4f}")
        
        weight_stats.append(max_val)

    print(f"-> Total Layers: {len(model_sd)}")
    print(f"-> Layers with NaN/Inf: {nan_params}")
    print(f"-> Layers with Large Weights (>10.0): {large_params}")

    print("\n--- 2. Analyzing Optimizer State (v_t instability) ---")
    # Optimizer state는 param_id (int)를 키로 가짐
    low_vt_count = 0
    zero_vt_count = 0
    
    for param_id, s in state.items():
        if 'exp_avg_sq' not in s:
            continue
            
        # v_t (variance estimate)
        v_t = s['exp_avg_sq'].float().numpy()
        m_t = s['exp_avg'].float().numpy()
        
        # v_t가 너무 작으면(0에 가까우면) update step이 폭발함
        # beta2가 낮으면 v_t가 0이 되는 경우가 많음 (Sparse gradient 등)
        min_v_t = np.min(v_t)
        
        if min_v_t < 1e-10:
            zero_vt_count += 1
        elif min_v_t < 1e-5:
            low_vt_count += 1
            
        v_t_stats.append(np.mean(v_t))
        
        # Update Step Size 추정 (Learning Rate 제외한 순수 비율)
        # step ~ m_t / (sqrt(v_t) + eps)
        # v_t가 작을 때 이 비율이 얼마나 커지는지 확인
        denom = np.sqrt(v_t) + 1e-8
        ratio = np.abs(m_t) / denom
        update_ratios.append(np.mean(ratio))

    if len(v_t_stats) > 0:
        print(f"-> Parameters with extremely low v_t (< 1e-5): {low_vt_count} (Risk of explosion)")
        print(f"-> Parameters with zero v_t: {zero_vt_count} (Unvisited sparse params)")
        print(f"-> Avg v_t value: {np.mean(v_t_stats):.6f}")
        print(f"-> Avg Update Ratio (|m|/sqrt(v)): {np.mean(update_ratios):.4f}")
        
        if np.mean(update_ratios) > 5.0:
            print("🚨 CRITICAL: Update ratios are very high. The model is likely taking huge steps.")

    # 시각화 (선택 사항)
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(np.log10(np.array(v_t_stats) + 1e-12), bins=50, color='orange', alpha=0.7)
        plt.title('Log10 Distribution of Average v_t (Variance)')
        plt.xlabel('Log10(v_t)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(np.array(update_ratios), bins=50, color='blue', alpha=0.7)
        plt.title('Distribution of Update Ratios (|m|/sqrt(v))')
        plt.xlabel('Ratio Size')
        plt.ylabel('Count')
        plt.yscale('log') # 큰 값들이 많을 수 있으므로 로그 스케일
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("\n📊 Histogram displayed. Check 'Log10(v_t)' for left-skewed spikes.")
    except Exception as e:
        print(f"\nCould not plot histogram: {e}")

# --- 실행 부분 ---
# 아래 경로를 실제 파일 경로로 수정하세요
if __name__ == "__main__":
    save_path = "experiments/train/iLRM_dl3dv_32input_256/ckpt_001000.pt"
    analyze_checkpoint(save_path)