import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const SystemMonitor: React.FC = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h4">System Monitor</Typography>
        <Typography>
          Real-time monitoring of all NextVerse modules
        </Typography>
      </CardContent>
    </Card>
  );
};

export default SystemMonitor;