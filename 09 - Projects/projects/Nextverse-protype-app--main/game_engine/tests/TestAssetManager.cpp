#include "gtest/gtest.h"
#include "AssetManager.hpp" // Class to be tested
// Panda3D types like NodePath, PT(Texture) are included via AssetManager.hpp

// Test fixture for AssetManager tests
class AssetManagerTest : public ::testing::Test {
protected:
    AssetManager assetManager;
};

// Test Case 1: Attempt to load a model that does not exist
TEST_F(AssetManagerTest, LoadNonExistentModel) {
    std::string nonExistentModelPath = "path/to/definitely_non_existent_model.egg";
    NodePath modelNode = assetManager.loadModel(nonExistentModelPath);
    ASSERT_TRUE(modelNode.is_empty()) << "Loading a non-existent model should result in an empty NodePath.";
}

// Test Case 2: Attempt to load a texture that does not exist
TEST_F(AssetManagerTest, LoadNonExistentTexture) {
    std::string nonExistentTexturePath = "path/to/definitely_non_existent_texture.png";
    PT(Texture) texturePtr = assetManager.getTexture(nonExistentTexturePath);
    ASSERT_EQ(texturePtr, nullptr) << "Loading a non-existent texture should result in nullptr.";
}

// Test Case 3: Test model caching behavior with failed loads
TEST_F(AssetManagerTest, CacheBehaviorForFailedModelLoad) {
    std::string nonExistentModelPath = "path/to/another_non_existent_model.egg";

    // First attempt
    NodePath modelNodeFirst = assetManager.loadModel(nonExistentModelPath);
    ASSERT_TRUE(modelNodeFirst.is_empty());

    // Second attempt (current AssetManager logic re-attempts load if cached is empty)
    NodePath modelNodeSecond = assetManager.loadModel(nonExistentModelPath);
    ASSERT_TRUE(modelNodeSecond.is_empty());
}

// Test Case 4: Test texture caching behavior with failed loads (similar to models)
TEST_F(AssetManagerTest, CacheBehaviorForFailedTextureLoad) {
    std::string nonExistentTexturePath = "path/to/another_non_existent_texture.png";

    // First attempt
    PT(Texture) textureFirst = assetManager.getTexture(nonExistentTexturePath);
    ASSERT_EQ(textureFirst, nullptr);

    // Second attempt (current AssetManager logic re-attempts load if cached is null)
    PT(Texture) textureSecond = assetManager.getTexture(nonExistentTexturePath);
    ASSERT_EQ(textureSecond, nullptr);
}
