

path = "D:\\AI_ML_Code\\PythonCode_Packages\\custom_ner\\custom_ner\\models\\bert_run_20241017_103925\\model_output\\bert_model/best_model/"
def parse_metrics_file(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Parsing individual classes and their metrics
        i = 2  # Skip the first two lines (header lines)
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith("micro avg") or line.startswith("macro avg") or line.startswith("weighted avg"):
                print(line.split())
                avg_type,avg, precision, recall, f1_score, support = line.split()
                metrics[avg_type+avg] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1-score": float(f1_score),
                    "support": int(support)
                }
            elif '=' in line:
                # Handle accuracy, eval_loss, etc.
                key, value = line.split('=', 1)
                metrics[key.strip()] = float(value.strip())
            else:
                # Handle other class metrics
                if not line:
                    i += 1
                    continue  # Skip empty lines
                parts = line.split()
                class_name = parts[0]
                print(parts)
                precision, recall, f1_score, support = float(parts[1]),float(parts[2]),float(parts[3]),int(parts[4])
                # precision, recall, f1_score, support = map(float, parts[1:4]) + [int(parts[4])]
                metrics[class_name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "support": support
                }
            i += 1

    return metrics

# Example usage
metrics_dict = parse_metrics_file(path+'eval_results.txt')
print(metrics_dict)
