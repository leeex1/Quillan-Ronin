#ifndef WEBBRIDGE_H
#define WEBBRIDGE_H

#include <QObject>
#include <QString>
#include <QJsonObject>
#include <QJsonDocument>
#include <QVariant>
#include <QTimer>

class WebBridge : public QObject
{
    Q_OBJECT

public:
    explicit WebBridge(QObject *parent = nullptr);

public slots:
    // Platform integration
    void showNotification(const QString &title, const QString &message);
    void setWindowTitle(const QString &title);
    void minimizeWindow();
    void maximizeWindow();
    void closeWindow();
    
    // Module communication
    QJsonObject callModule(const QString &moduleName, const QString &method, const QJsonObject &params);
    void registerModuleCallback(const QString &moduleName, const QString &event);
    
    // Performance monitoring
    QJsonObject getPerformanceMetrics();
    void startPerformanceMonitoring();
    void stopPerformanceMonitoring();
    
    // System integration
    QString getClipboard();
    void setClipboard(const QString &text);
    void openExternalUrl(const QString &url);
    void openFileDialog(const QString &filter = "");
    void saveFileDialog(const QString &defaultName = "", const QString &filter = "");

signals:
    // Signals that can be emitted to JavaScript
    void moduleEvent(const QString &moduleName, const QString &event, const QJsonObject &data);
    void performanceUpdate(const QJsonObject &metrics);
    void systemNotification(const QString &type, const QString &message);

private slots:
    void updatePerformanceMetrics();

private:
    QTimer *m_performanceTimer;
    bool m_performanceMonitoring;
    
    // Mock module registry for demonstration
    QMap<QString, QJsonObject> m_moduleRegistry;
    
    void initializeModules();
    QJsonObject simulateModuleCall(const QString &moduleName, const QString &method, const QJsonObject &params);
};

#endif // WEBBRIDGE_H
