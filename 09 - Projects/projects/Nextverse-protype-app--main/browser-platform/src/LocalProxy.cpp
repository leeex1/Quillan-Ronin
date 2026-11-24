#include "LocalProxy.h"
#include <QStandardPaths>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonObject>
#include <QJsonDocument>
#include <QSysInfo>
#include <QCoreApplication>
#include <QSettings>
#include <QDebug>

LocalProxy::LocalProxy(QObject *parent)
    : QObject(parent)
{
    // Initialize paths
    m_appDataPath = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    m_assetsPath = QCoreApplication::applicationDirPath() + "/assets";
    
    // Ensure directories exist
    QDir().mkpath(m_appDataPath);
    QDir().mkpath(m_assetsPath);
}

QString LocalProxy::readFile(const QString &path)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        qWarning() << "Unsafe path access attempted:" << path;
        return QString();
    }
    
    QFile file(safePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Failed to open file for reading:" << safePath;
        return QString();
    }
    
    return file.readAll();
}

bool LocalProxy::writeFile(const QString &path, const QString &content)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        qWarning() << "Unsafe path access attempted:" << path;
        return false;
    }
    
    // Ensure directory exists
    QFileInfo fileInfo(safePath);
    QDir().mkpath(fileInfo.absolutePath());
    
    QFile file(safePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "Failed to open file for writing:" << safePath;
        return false;
    }
    
    file.write(content.toUtf8());
    return true;
}

QStringList LocalProxy::listDirectory(const QString &path)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        qWarning() << "Unsafe path access attempted:" << path;
        return QStringList();
    }
    
    QDir dir(safePath);
    if (!dir.exists()) {
        return QStringList();
    }
    
    return dir.entryList(QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot);
}

bool LocalProxy::createDirectory(const QString &path)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        qWarning() << "Unsafe path access attempted:" << path;
        return false;
    }
    
    return QDir().mkpath(safePath);
}

bool LocalProxy::deleteFile(const QString &path)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        qWarning() << "Unsafe path access attempted:" << path;
        return false;
    }
    
    return QFile::remove(safePath);
}

bool LocalProxy::fileExists(const QString &path)
{
    QString safePath = sanitizePath(path);
    if (!isPathSafe(safePath)) {
        return false;
    }
    
    return QFile::exists(safePath);
}

QJsonObject LocalProxy::getConfig()
{
    QSettings settings;
    QJsonObject config;
    
    // Read all settings into JSON object
    for (const QString &key : settings.allKeys()) {
        config[key] = settings.value(key).toString();
    }
    
    return config;
}

bool LocalProxy::saveConfig(const QJsonObject &config)
{
    QSettings settings;
    
    // Write JSON object to settings
    for (auto it = config.begin(); it != config.end(); ++it) {
        settings.setValue(it.key(), it.value().toString());
    }
    
    settings.sync();
    return settings.status() == QSettings::NoError;
}

QString LocalProxy::getAssetPath(const QString &assetName)
{
    QString assetPath = m_assetsPath + "/" + assetName;
    if (QFile::exists(assetPath)) {
        return assetPath;
    }
    return QString();
}

QStringList LocalProxy::getAvailableAssets()
{
    QDir assetsDir(m_assetsPath);
    return assetsDir.entryList(QDir::Files);
}

QJsonObject LocalProxy::getSystemInfo()
{
    QJsonObject info;
    
    info["platform"] = QSysInfo::productType();
    info["version"] = QSysInfo::productVersion();
    info["architecture"] = QSysInfo::currentCpuArchitecture();
    info["kernelType"] = QSysInfo::kernelType();
    info["kernelVersion"] = QSysInfo::kernelVersion();
    info["prettyName"] = QSysInfo::prettyProductName();
    info["appDataPath"] = m_appDataPath;
    info["assetsPath"] = m_assetsPath;
    
    return info;
}

QString LocalProxy::getLocalData(const QString &key)
{
    QSettings settings("NextVerse", "LocalData");
    return settings.value(key).toString();
}

bool LocalProxy::setLocalData(const QString &key, const QString &value)
{
    QSettings settings("NextVerse", "LocalData");
    settings.setValue(key, value);
    settings.sync();
    return settings.status() == QSettings::NoError;
}

bool LocalProxy::removeLocalData(const QString &key)
{
    QSettings settings("NextVerse", "LocalData");
    settings.remove(key);
    settings.sync();
    return settings.status() == QSettings::NoError;
}

QStringList LocalProxy::getLocalDataKeys()
{
    QSettings settings("NextVerse", "LocalData");
    return settings.allKeys();
}

QString LocalProxy::getAppDataPath()
{
    return m_appDataPath;
}

QString LocalProxy::getAssetsPath()
{
    return m_assetsPath;
}

QString LocalProxy::sanitizePath(const QString &path)
{
    // Convert to absolute path relative to app data directory
    if (path.startsWith("/") || path.contains("..")) {
        // For absolute paths or paths with .., restrict to app data directory
        return m_appDataPath + "/" + QFileInfo(path).fileName();
    }
    
    // For relative paths, prepend app data directory
    return m_appDataPath + "/" + path;
}

bool LocalProxy::isPathSafe(const QString &path)
{
    // Ensure path is within allowed directories
    QString canonicalPath = QFileInfo(path).canonicalFilePath();
    
    return canonicalPath.startsWith(m_appDataPath) || 
           canonicalPath.startsWith(m_assetsPath) ||
           canonicalPath.startsWith(QCoreApplication::applicationDirPath());
}
