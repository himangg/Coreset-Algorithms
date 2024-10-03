# Coreset Construction Algorithms for Projective Clustering

This repository implements algorithms for constructing accurate coresets in projective clustering, specifically focusing on one-segment queries. The algorithms leverage the Carathéodory theorem to efficiently select a minimal set of representative points while maintaining the essential characteristics of the original dataset.

## Algorithms

### 1. Coreset Construction
- **Description**: This algorithm constructs a coreset by selecting representative points that approximate the original dataset.
- **Key Features**:
  - Achieves 100% accuracy with the complete dataset.
  - Efficiently handles one-segment queries.

### 2. Modification for One-Segment Queries
- **Description**: This modification tailors the core set construction specifically for one-segment queries in projective clustering.
- **Key Features**:
  - Adapts the general coreset construction method to enhance performance for one-segment queries.
  - Improves computational efficiency while preserving accuracy.

### 3. Carathéodory Theorem Implementation
- **Description**: Utilizes the Carathéodory theorem to select a minimal set of representative points.
- **Key Features**:
  - Reduces the complexity of the dataset by summarizing points effectively.
  - Maintains the essential characteristics of the original dataset.
