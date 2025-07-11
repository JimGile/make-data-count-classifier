That's an excellent question, Jim, and it brings us to the deployment phase for your Kaggle submission!

For a project like this, especially with fine-tuning an LLM, the best practice on Kaggle is almost always to use **two separate notebooks**:

1.  **Training Notebook:** Where you train your model and save the fine-tuned LoRA adapters.
2.  **Inference Notebook:** Where you load the saved adapters, load the base model, perform inference on the test data, and generate your `submission.csv` file.

Here's why this is the recommended approach and how to set it up:

### Why Two Notebooks?

1.  **Kaggle Time Limits:** Training can be very time-consuming. Kaggle notebooks have runtime limits (e.g., 9-12 hours for GPU notebooks). Separating training allows you to iterate on inference faster without re-running the long training process every time.
2.  **Resource Management:** Training often requires more GPU memory and compute. Inference can sometimes be done with fewer resources or even on a CPU if necessary (though GPU is always faster for LLMs).
3.  **Clearer Workflow:** It separates the concerns of model development from model deployment.
4.  **Reproducibility:** Your submission notebook is clean and focused solely on generating predictions.

### Step-by-Step Guide

#### Notebook 1: Training the Model

This is essentially the notebook you've been working on.

1.  **Ensure `trainer.save_model()` is called:**
    You already have this:
    ```python
    trainer.save_model(os.path.join(output_dir, "final_model"))
    ```
    Make sure `output_dir` is a path within your notebook's writable directory (e.g., `./results`).

2.  **Run the Training Notebook:** Execute all cells in your training notebook.
3.  **Save the Output as a Kaggle Dataset:**
    *   Once the training run is complete and successful, go to the "Output" tab of your Kaggle notebook.
    *   You should see the `results/final_model` directory (or whatever `output_dir` you used) listed there.
    *   Click the "Add Data" button (or similar option) next to this output. Kaggle will prompt you to "Save Version" of the notebook.
    *   After saving, Kaggle will process the output. This output (your `final_model` directory containing the LoRA adapters) will then become a "Notebook Output" dataset.

#### Notebook 2: Inference for Submission

This will be a new notebook.

1.  **Create a New Notebook:** Start a fresh Kaggle notebook.
2.  **Add Your Trained Model as Input:**
    *   In the new inference notebook, click on "Add Data" (usually on the right sidebar).
    *   Go to the "Notebooks" tab.
    *   Search for your *training notebook's name*.
    *   Select the output of your training notebook (it will show up as a dataset).
    *   Click "Add".

    Kaggle will mount this output as an input dataset. It will typically be mounted under `/kaggle/input/<your-training-notebook-slug>/<your-output-directory-name>`.
    For example, if your training notebook was named `llm-finetuning-train` and your output directory was `results/final_model`, it might be mounted as `/kaggle/input/llm-finetuning-train/results/final_model`.

3.  **Load the Base Model and Tokenizer:**
    You need to load the *same base model* (e.g., Qwen 0.5B/1.8B) and tokenizer you used for training. Ensure you use the same `BitsAndBytesConfig` if you used 4-bit quantization during training.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch
    import os

    # Define the base model ID (same as training)
    base_model_id = "Qwen/Qwen1.5-0.5B-Chat" # Or "Qwen/Qwen1.5-1.8B-Chat"

    # Configure quantization (MUST match training if you used it)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16 if you used fp16
        bnb_4bit_use_double_quant=False,
    )

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto", # Use "auto" for efficient device placement
        trust_remote_code=True
    )
    model.config.use_cache = True # Enable cache for faster inference

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Or whatever you set during training
    tokenizer.padding_side = "right" # Or "left" if you used it for training
    ```

4.  **Load the Fine-tuned LoRA Adapters:**
    This is where you load your trained model from the input path.

    ```python
    # Define the path to your saved LoRA adapters
    # Adjust this path based on how Kaggle mounted your training notebook's output
    # You can find the exact path by looking at the "Input" section in your notebook's sidebar
    # Example: If your training notebook was 'my-llm-train' and output dir was 'results/final_model'
    lora_model_path = "/kaggle/input/my-llm-train-notebook-output/results/final_model" # <--- ADJUST THIS PATH

    # Load the PeftModel (LoRA adapters) on top of the base model
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.eval() # Set model to evaluation mode
    print("Fine-tuned model loaded successfully for inference!")
    ```

5.  **Prepare Test Data and Run Inference:**
    *   Load the Kaggle test dataset (usually from `/kaggle/input/data-citation-classification/test.csv`).
    *   Apply the *same preprocessing and `format_example` logic* to the test data to create the prompts.
    *   Iterate through the test data, generate predictions for each example.

    ```python
    # Example inference loop (simplified)
    import pandas as pd
    from tqdm.auto import tqdm # For progress bar

    # Load test data (adjust path as needed)
    test_df = pd.read_csv("/kaggle/input/data-citation-classification/test.csv")

    # You'll need to adapt your data loading and preprocessing for the test set
    # For example, if your test_df has 'article_doi', 'abstract', 'citation_context', 'dataset_id'
    # You'd create prompts similar to your format_example function, but without the 'label'

    # Example of a single inference:
    def generate_prediction(example_row):
        # Construct the prompt using the same format as your training's format_example
        # but without the assistant's response (the 'label')
        messages = [
            {"role": "system", "content": "You are an expert assistant for classifying research data citations."},
            {"role": "user", "content": (
                f"Given the following article context and a specific data citation, classify if the data was generated as 'Primary' (newly generated for this study) or 'Secondary' (reused from existing records).\n\n"
                f"Article Abstract: {example_row['abstract']}\n"
                f"Article DOI: {example_row['article_doi']}\n"
                f"Data Citation Context: {example_row['citation_context']}\n"
                f"Dataset ID: {example_row['dataset_id']}\n\n"
                f"Classification:"
            )}
        ]
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # add_generation_prompt=True for inference

        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to("cuda") # Ensure max_length matches training

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10, # Only generate a few tokens for "Primary" or "Secondary"
                do_sample=False,   # For deterministic output
                pad_token_id=tokenizer.eos_token_id # Important for generation
            )
        
        # Decode the generated text and extract the classification
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process to get just "Primary" or "Secondary"
        # This will depend on how your model generates. You might need to parse the last part.
        # Example: "Classification: Primary" -> "Primary"
        if "Primary" in generated_text:
            return "Primary"
        elif "Secondary" in generated_text:
            return "Secondary"
        else:
            return "Unknown" # Handle cases where it doesn't classify clearly

    predictions = []
    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pred = generate_prediction(row)
        predictions.append(pred)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'], # Assuming your test_df has an 'id' column
        'label': predictions
    })
    ```

6.  **Generate Submission File:**
    ```python
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")
    ```

7.  **Submit the Inference Notebook:**
    *   Save your inference notebook.
    *   Click "Submit" (or "Save Version" and then "Submit"). Kaggle will run this notebook, generate `submission.csv`, and evaluate it against the competition's test set.

This two-notebook workflow is the standard and most robust way to manage your LLM fine-tuning and submission on Kaggle, Jim. Good luck!