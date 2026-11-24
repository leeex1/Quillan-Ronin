#ifndef ASSETMANAGER_HPP
#define ASSETMANAGER_HPP

#include <string>
#include <iostream>
#include <map>

#include "nodePath.h"
#include "loader.h"
#include "filename.h"
#include "texturePool.h" // For TexturePool
#include "texture.h"     // For Texture and PT(Texture)

class AssetManager {
public:
    AssetManager();
    ~AssetManager();

    NodePath loadModel(const std::string& filePath);
    PT(Texture) getTexture(const std::string& texturePath); // Returns PT(Texture)

    void listLoadedAssets() const;

private:
    std::map<std::string, NodePath> model_cache_;
    std::map<std::string, PT(Texture)> texture_cache_; // Cache PT(Texture)
};

#endif // ASSETMANAGER_HPP
