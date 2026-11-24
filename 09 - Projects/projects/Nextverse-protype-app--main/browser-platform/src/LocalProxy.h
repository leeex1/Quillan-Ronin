#ifndef LOCALPROXY_H
#define LOCALPROXY_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QDir>
#include <QStandardPaths>

class LocalProxy : public QObject
{
    Q_OBJECT

public:
    explicit LocalProxy(QObject *parent = nullptr);

public slots:
    // File system operations
    QString readFile(const QString &path);
    bool writeFile(const QString &path, const QString &content);
    QStringList listDirectory(const QString &path);
    bool createDirectory(const QString &path);
    bool deleteFile(const QString &path);
    bool fileExists(const QString &path);
    
    // Configuration management
    QJsonObject getConfig();
    bool saveConfig(const QJsonObject &config);
    
    // Asset management
    QString getAssetPath(const QString &assetName);
    QStringList getAvailableAssets();
    
    // System information
    QJsonObject getSystemInfo();
    
    // Local storage operations (bypass CORS)
    QString getLocalData(const QString &key);
    bool setLocalData(const QString &key, const QString &value);
    bool removeLocalData(const QString &key);
    QStringList getLocalDataKeys();

private:
    QString getAppDataPath();
    QString getAssetsPath();
    QString sanitizePath(const QString &path);
    bool isPathSafe(const QString &path);
    
    QString m_appDataPath;
    QString m_assetsPath;
};

#endif // LOCALPROXY_H
