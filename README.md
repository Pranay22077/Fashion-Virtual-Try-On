# Virtual Try-On and Fashion Recommendation System
<p align="center">
  <img src="composite_images_semifinal/composite_04269_12171_00.jpg" width="250"/>
  <img src="composite_images_semifinal/composite_02278_11237_00.jpg" width="250"/>
  <img src="composite_images_semifinal/composite_03429_07276_00.jpg" width="250"/>
   <img src="composite_images_semifinal/composite_07212_11330_00.jpg" width="250"/>
</p>

<p align="center"><i>Virtual Try-On Results (Generated via VITON-HD)</i></p>


This project presents a sophisticated system that integrates a high-resolution virtual try-on (VTON) model with an intelligent, multi-layered fashion recommendation engine. The system allows users to visualize themselves in new clothing items and receive personalized, context-aware fashion recommendations. The core of the project leverages the state-of-the-art **VITON-HD** model for virtual try-on and a novel hybrid architecture combining **CLIP** embeddings with an **LLM reasoning layer** for recommendations.

---

## System Architecture

The system is architected around three core components: a universal pre-processing pipeline, a virtual try-on module, and a recommendation engine.

### Data Pre-processing Pipeline

To make the system compatible with any standard user photo, a three-step automated pipeline was developed to generate the data-rich inputs required by the VITON-HD model.

1.  **Cloth Mask Generation**: The `rembg` library is used to remove the background from a garment image, creating a clean, binary mask. This ensures that only the clothing item is warped and applied to the user.
2.  **Human Segmentation Map**: A specialized human parsing model (`image-parse-v3`) analyzes the user's photo to generate a color-coded map of body parts (e.g., arms, torso, neck). This map guides the VTON model in correctly placing clothing features like sleeves and necklines.
3.  **Pose Information Generation**: **OpenPose** is used to extract precise `(x, y)` coordinates of body joints (`openpose-json`) and a visual skeleton map (`openpose-img`). This critical step allows the model to accurately deform the garment to match the person's specific pose.

### Virtual Try-On Module

The virtual try-on capability is powered by **VITON-HD**, a high-resolution framework that operates in two main stages.

1.  **Geometric Matching Module (GMM)**: This module aligns the target clothing item with the person's body shape and pose. It uses a U-Net-like architecture to predict the parameters for a **Thin Plate Spline (TPS)** transformation. The TPS warp is applied to the cloth image and its mask, creating a deformed garment that fits the user's posture.
2.  **Try-On Module (TOM)**: This module renders the final photorealistic image. It takes the warped cloth from the GMM and the person's representation as input. Its key innovation is the **ALIgnment-Aware Segment (ALIAS) Normalization** layer, which applies learned, spatially-aware affine transformations based on the human segmentation map. This preserves fine details in the original image while intelligently synthesizing textures where the garment is placed, preventing artifacts and producing a seamless blend.

### Recommendation Engine

The recommendation engine employs a hybrid architecture designed for computational efficiency and nuanced, context-aware suggestions.

1.  **Deep Learning Backbone**: The vision encoder from OpenAI's **CLIP** (`clip-vit-base-patch32`) is used to generate semantic vector embeddings for all clothing items in the catalog. These embeddings are stored in a **FAISS** index, which enables highly efficient, large-scale similarity searches.
2.  **LLM Reasoning Layer**: A **Gemini-powered LLM** acts as a high-level reasoning engine. It analyzes the user's input (photo and optional text prompt) to determine the optimal search strategy. Based on its analysis, it can trigger:
    * **Similar Search**: Find items visually similar to what the user is wearing.
    * **Complementary Search**: Suggest items that pair well with the user's current outfit.
    * **Evaluation-Only**: Score the user's current outfit based on style and trends.

---

## Development and Methodology

The final architecture was the result of systematic experimentation and adaptation in response to technical challenges.

### Virtual Try-On Approach

The initial goal was to **fine-tune the VITON-HD model** on a custom dataset by unfreezing and training approximately 97 million parameters, primarily in the ALIAS Generator. However, after 19 hours of training, the model exhibited a persistent "ghosting" effect and the loss function failed to converge significantly.

An alternative, **OOTDiffusion**, was briefly explored but was abandoned due to complex dependency conflicts and environment setup errors on the Kaggle platform.

The final, pragmatic approach was to use the **original, pre-trained VITON-HD model** for direct inference. This successfully demonstrated the end-to-end pipeline and served as a powerful benchmark, highlighting the significant computational resources required for effective fine-tuning.

### Recommendation Engine Approach

Several initial strategies for the recommendation engine were attempted:
* A **Siamese network** failed due to a lack of user history and metadata in the dataset.
* A **FashionBERT-style transformer** was computationally infeasible, facing GPU memory constraints and poor convergence.
* **Graph Neural Network (GNN)** models were unsuitable as they required explicit outfit pairing data, which was not available.

These challenges led to the final **hybrid CLIP + LLM architecture**, which proved robust, computationally feasible, and effective at overcoming the dataset's limitations.

---

## Technical Challenges and Innovations

Development was marked by several significant technical blockers, which led to innovative solutions.

**Blockers Encountered**:

* **Attribute Extraction**: Tools like **DeepFace** and **InsightFace** for skin tone and gender analysis produced inconsistent results and suffered from initialization errors.
* **Logo Detection**: CLIP-based text search, OpenCV template matching, and pre-trained YOLOv8 models all failed to reliably detect small or varied logos in the dataset.
* **Computational Cost**: Pose generation with **OpenPose** and body parsing were computationally intensive and sometimes failed on complex images.
* **Model Fine-Tuning**: Fine-tuning VITON-HD was plagued by `RuntimeError: size mismatch` errors, impractically long training times (8-9 hours for a few epochs), Kaggle's session limits, and a persistent "ghosting" effect in the output.
* **Environment Setup**: Newer models like **OOTDiffusion** had complex dependencies that conflicted with the Kaggle environment. The pre-trained VITON-HD required manual code updates to resolve pathing issues and replace the deprecated `torchgeometry` library with `kornia`.

**Key Innovations**:

* **Hybrid LLM Architecture**: The primary innovation was integrating an LLM reasoning layer with a deep learning backbone. This bridged the gap between low-level visual embeddings and high-level fashion concepts, allowing the system to reason about user intent, trends, and outfit compatibility without explicit metadata.
* **Compatibility-Aware Search**: The LLM layer enabled a shift from simple visual similarity to more sophisticated "complementary" recommendations, a crucial feature for a practical fashion tool.

---

## Results

Despite the challenges with fine-tuning, the project successfully implemented a full, end-to-end pipeline using the pre-trained VITON-HD model. The system capably generates high-resolution virtual try-on images from arbitrary user and clothing photos. The recommendation engine effectively provides dynamic, context-aware suggestions, demonstrating the power of its hybrid architecture. The final system serves as a valuable proof-of-concept and a robust benchmark for future work in this domain.

---

## References

* **VITON-HD**:
    * Paper: [https://arxiv.org/pdf/2103.16874](https://arxiv.org/pdf/2103.16874)
    * GitHub: [https://github.com/shadow2496/VITON-HD](https://github.com/shadow2496/VITON-HD)
* **OOTDiffusion**:
    * Paper: [https://arxiv.org/pdf/2403.01779](https://arxiv.org/pdf/2403.01779)
    * GitHub: [https://github.com/levihsu/OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
* **Other Relevant Implementations**:
    * [https://github.com/sonu275981/Fashion-Recommender-system](https://github.com/sonu275981/Fashion-Recommender-system)
    * [https://github.com/jjoej15/outfit-detect-recs](https://github.com/jjoej15/outfit-detect-recs)
    * [https://github.com/knowrohit/Fashion-Rec-Sys](https://github.com/knowrohit/Fashion-Rec-Sys)
    * [https://github.com/sakshamarora97/outfit-compatibility-scoring](https://github.com/sakshamarora97/outfit-compatibility-scoring)

---

## Authors

This project was created by **Pranay Gadh** and **Shanjan Makkar** from **Delhi Technological University**.
