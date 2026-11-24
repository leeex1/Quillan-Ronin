#include "AssetManager.hpp"
// Panda3D Loader and TexturePool are globally accessible
// Other Panda3D headers like Filename, Texture are included via AssetManager.hpp

AssetManager::AssetManager() {
    std::cout << "AssetManager created." << std::endl;
}

AssetManager::~AssetManager() {
    std::cout << "AssetManager destroyed." << std::endl;
}

// loadModel remains the same as in the previous step, with slightly less verbose logging
NodePath AssetManager::loadModel(const std::string& filePath) {
    auto it = model_cache_.find(filePath);
    if (it != model_cache_.end()) {
        if (!it->second.is_empty()) { // Check if cached NodePath is valid
            // std::cout << "  AssetManager: Model '" << filePath << "' found in cache and is valid." << std::endl; // Less verbose
            return it->second;
        } else {
            // std::cout << "  AssetManager: Model '" << filePath << "' found in cache but was invalid (empty NodePath). Attempting reload." << std::endl; // Less verbose
            // Fall through to load again.
        }
    }

    // std::cout << "  AssetManager: Loading model from '" << filePath << "' using Panda3D Loader..." << std::endl; // Less verbose
    NodePath model_node = Loader::get_global_ptr()->load_sync(Filename::from_os_specific(filePath));

    if (model_node.is_empty()) {
        std::cerr << "  AssetManager: Error! Failed to load model: " << filePath << std::endl;
        model_cache_[filePath] = model_node; // Cache the empty (invalid) NodePath
        return NodePath(); // Return an empty NodePath
    }

    // std::cout << "  AssetManager: Model '" << filePath << "' loaded successfully. Panda Name: " << model_node.get_name() << std::endl; // Less verbose
    model_cache_[filePath] = model_node;
    return model_node;
}

PT(Texture) AssetManager::getTexture(const std::string& texturePath) {
    auto it = texture_cache_.find(texturePath);
    if (it != texture_cache_.end()) {
        if (it->second != nullptr) { // Check if cached PT(Texture) is valid (not nullptr)
            // std::cout << "  AssetManager: Texture '" << texturePath << "' found in cache and is valid." << std::endl; // Less verbose
            return it->second;
        } else {
             // std::cout << "  AssetManager: Texture '" << texturePath << "' found in cache but was invalid (nullptr). Attempting reload." << std::endl; // Less verbose
             // Fall through to load again.
        }
    }

    std::cout << "  AssetManager: Loading texture '" << texturePath << "' using Panda3D TexturePool..." << std::endl;
    PT(Texture) texture_ptr = TexturePool::load_texture(Filename::from_os_specific(texturePath));

    if (texture_ptr == nullptr) { // TexturePool::load_texture returns nullptr on failure
        std::cerr << "  AssetManager: Error! Failed to load texture: " << texturePath << std::endl;
        texture_cache_[texturePath] = nullptr; // Cache the failure (nullptr)
        return nullptr;
    }

    std::cout << "  AssetManager: Texture '" << texturePath << "' loaded successfully. "
              << "Dimensions: " << texture_ptr->get_x_size() << "x" << texture_ptr->get_y_size()
              << ", Components: " << texture_ptr->get_num_components() << std::endl;
    texture_cache_[texturePath] = texture_ptr;
    return texture_ptr;
}

void AssetManager::listLoadedAssets() const {
    std::cout << "  AssetManager: Listing loaded assets..." << std::endl;
    if (model_cache_.empty()) {
        std::cout << "    Models: None" << std::endl;
    } else {
        std::cout << "    Models (" << model_cache_.size() << "):" << std::endl;
        for (const auto& pair : model_cache_) {
            std::cout << "      - Path: " << pair.first
                      << ", Cached NodePath is_empty: " << (pair.second.is_empty() ? "Yes (load failed)" : "No")
                      << (pair.second.is_empty() ? "" : ", Panda Name: " + pair.second.get_name()) << std::endl;
        }
    }
    if (texture_cache_.empty()) {
         std::cout << "    Textures: None" << std::endl;
    } else {
        std::cout << "    Textures (" << texture_cache_.size() << "):" << std::endl;
        for (const auto& pair : texture_cache_) {
            if (pair.second != nullptr) { // Check if the PT(Texture) is valid
                std::cout << "      - Path: " << pair.first
                          << ", Loaded: Yes, Dim: " << pair.second->get_x_size() << "x" << pair.second->get_y_size()
                          << ", Comp: " << pair.second->get_num_components() << std::endl;
            } else {
                std::cout << "      - Path: " << pair.first << ", Loaded: No (load failed)" << std::endl;
            }
        }
    }
}
