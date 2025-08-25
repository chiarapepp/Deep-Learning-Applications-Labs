#!/bin/bash

# =============================================================================
# ESPERIMENTI SISTEMATICI ADATTATI AL TUO CODICE
# Ispirato agli esperimenti del tuo amico ma adattato alla tua interfaccia
# =============================================================================

echo "Running systematic experiments (inspired by friend's design)..."
mkdir -p Models logs results

# Parametri base
lr=0.001
epochs=50
batch_size=128
dataset="MNIST"

# =============================================================================
# DEPTH STUDY: Confronto MLP vs ResMLP a diverse profondità
# =============================================================================

echo "=== DEPTH STUDY ==="

# Depth 10 → num_blocks = 5 (circa)
depths=(
    "128,64,32,16,8,4,2,1,1,1"          # MLP 10 layer
    "5"                                  # ResMLP 5 blocks
)

# Depth 20 → num_blocks = 10 (circa)  
depths_20=(
    "128,128,64,64,32,32,16,16,8,8,4,4,2,2,1,1,1,1,1,1"  # MLP 20 layer
    "10"                                                   # ResMLP 10 blocks
)

# Depth 40 → num_blocks = 20 (circa)
depths_40=(
    # Per MLP molto profondo, usa hidden_sizes ripetute
    "20"  # ResMLP 20 blocks (troppo lungo fare MLP a 40 layer)
)

# =============================================================================
# WIDTH STUDY: Diverse larghezze
# =============================================================================

widths=(32 64 128)

# =============================================================================
# ESPERIMENTI SISTEMATICI
# =============================================================================

experiment_counter=0

for width in "${widths[@]}"; do
    echo "Testing width: $width"
    
    # SHALLOW NETWORKS (depth ~10)
    echo "  Shallow networks (~10 layers)"
    
    # MLP shallow con/senza normalization, con/senza scheduler
    for normalization in "" "--normalization"; do
        for scheduler in "" "--use_scheduler"; do
            experiment_counter=$((experiment_counter + 1))
            echo "    Experiment $experiment_counter: MLP shallow, width=$width"
            
            if [ $width -eq 32 ]; then
                hidden_sizes="$width $((width/2)) $((width/4)) $((width/8))"
            elif [ $width -eq 64 ]; then  
                hidden_sizes="$width $((width/2)) $((width/4)) $((width/8)) $((width/16))"
            else
                hidden_sizes="$width $((width/2)) $((width/4)) $((width/8)) $((width/16)) $((width/32))"
            fi
            
            python main_ex1.py \
                --model mlp \
                --dataset $dataset \
                --hidden_sizes $hidden_sizes \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                $normalization \
                $scheduler \
                --use_wandb \
                2>&1 | tee results/exp${experiment_counter}_mlp_shallow_w${width}_norm${normalization}_sched${scheduler}.log &
                
            sleep 2
        done
    done
    
    # ResMLP shallow con/senza normalization, con/senza scheduler  
    for normalization in "" "--normalization"; do
        for scheduler in "" "--use_scheduler"; do
            experiment_counter=$((experiment_counter + 1))
            echo "    Experiment $experiment_counter: ResMLP shallow, width=$width"
            
            python main_ex1.py \
                --model resmlp \
                --dataset $dataset \
                --hidden_dim $width \
                --num_blocks 3 \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                $normalization \
                $scheduler \
                --use_wandb \
                2>&1 | tee results/exp${experiment_counter}_resmlp_shallow_w${width}_norm${normalization}_sched${scheduler}.log &
                
            sleep 2
        done
    done
    
    wait  # Aspetta che finiscano i shallow prima di continuare
    
    # MEDIUM NETWORKS (depth ~20)
    echo "  Medium networks (~20 layers)"
    
    # Solo ResMLP per medium depth (MLP troppo profondo)
    for normalization in "" "--normalization"; do
        for scheduler in "" "--use_scheduler"; do
            experiment_counter=$((experiment_counter + 1))
            echo "    Experiment $experiment_counter: ResMLP medium, width=$width"
            
            python main_ex1.py \
                --model resmlp \
                --dataset $dataset \
                --hidden_dim $width \
                --num_blocks 6 \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                $normalization \
                $scheduler \
                --use_wandb \
                2>&1 | tee results/exp${experiment_counter}_resmlp_medium_w${width}_norm${normalization}_sched${scheduler}.log &
                
            sleep 2
        done
    done
    
    wait
    
    # DEEP NETWORKS (depth ~40)  
    echo "  Deep networks (~40 layers)"
    
    # Solo ResMLP per deep (MLP impossibile)
    for normalization in "" "--normalization"; do
        for scheduler in "" "--use_scheduler"; do
            experiment_counter=$((experiment_counter + 1))
            echo "    Experiment $experiment_counter: ResMLP deep, width=$width"
            
            python main_ex1.py \
                --model resmlp \
                --dataset $dataset \
                --hidden_dim $width \
                --num_blocks 12 \
                --lr $lr \
                --epochs $epochs \
                --batch_size $batch_size \
                $normalization \
                $scheduler \
                --use_wandb \
                2>&1 | tee results/exp${experiment_counter}_resmlp_deep_w${width}_norm${normalization}_sched${scheduler}.log &
                
            sleep 2
        done
    done
    
    wait
done

# =============================================================================
# GRADIENT DEGRADATION STUDY
# Confronto diretto MLP vs ResMLP su reti che MLP può ancora gestire
# =============================================================================

echo "=== GRADIENT DEGRADATION STUDY ==="

# Test a profondità crescente fino a quando MLP fallisce
test_depths=(2 4 6 8)

for depth in "${test_depths[@]}"; do
    echo "Testing degradation at depth: $depth"
    
    # Crea hidden_sizes per MLP
    hidden_sizes=""
    for i in $(seq 1 $depth); do
        hidden_sizes="$hidden_sizes 128"
    done
    hidden_sizes=$(echo $hidden_sizes | xargs)  # trim spaces
    
    experiment_counter=$((experiment_counter + 1))
    echo "  Experiment $experiment_counter: MLP depth $depth"
    
    # MLP
    python main_ex1.py \
        --model mlp \
        --dataset $dataset \
        --hidden_sizes $hidden_sizes \
        --lr $lr \
        --epochs 10 \
        --batch_size $batch_size \
        --use_wandb \
        2>&1 | tee results/degradation_mlp_depth${depth}.log &
    
    # ResMLP equivalente
    experiment_counter=$((experiment_counter + 1))  
    echo "  Experiment $experiment_counter: ResMLP depth $depth"
    
    python main_ex1.py \
        --model resmlp \
        --dataset $dataset \
        --hidden_dim 128 \
        --num_blocks $depth \
        --lr $lr \
        --epochs 10 \
        --batch_size $batch_size \
        --use_wandb \
        2>&1 | tee results/degradation_resmlp_depth${depth}.log &
    
    wait
    sleep 5
done

# =============================================================================
# SUMMARY
# =============================================================================

echo "=== EXPERIMENT SUMMARY ==="
echo "Total experiments run: $experiment_counter"
echo ""
echo "Results saved in:"
echo "- Logs: ./results/"
echo "- Models: ./Models/"  
echo "- W&B dashboard for metrics"
echo ""
echo "Key comparisons to analyze:"
echo "1. MLP vs ResMLP at same approximate depth"
echo "2. Effect of normalization on deep networks"
echo "3. Effect of scheduler on convergence"
echo "4. Width vs depth tradeoffs"
echo "5. Gradient degradation patterns"

# Genera script di analisi
cat > analyze_results.py << 'EOF'
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_logs():
    """Parse log files to extract key metrics"""
    results = []
    
    for filename in os.listdir('results/'):
        if filename.endswith('.log'):
            # Extract parameters from filename
            params = {}
            if 'mlp' in filename:
                params['model'] = 'MLP'
            elif 'resmlp' in filename:
                params['model'] = 'ResMLP'
            
            # Extract other parameters
            width_match = re.search(r'w(\d+)', filename)
            if width_match:
                params['width'] = int(width_match.group(1))
            
            depth_match = re.search(r'depth(\d+)', filename)
            if depth_match:
                params['depth'] = int(depth_match.group(1))
            
            params['normalization'] = '--normalization' in filename
            params['scheduler'] = '--use_scheduler' in filename
            
            # TODO: Parse actual metrics from log content
            # For now just store parameters
            results.append(params)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = parse_logs()
    print("Experiment parameters summary:")
    print(df.groupby(['model', 'width']).size())
EOF

echo "Analysis script created: analyze_results.py"
echo "Run: python analyze_results.py"