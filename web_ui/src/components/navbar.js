import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { AppBar,Typography,Toolbar } from '@material-ui/core';

const useStyles = makeStyles(theme => ({
    root: {
      flexGrow: 1,
    },
    paper: {
      padding: theme.spacing(2),
      textAlign: 'center',
      color: theme.palette.text.secondary,
    },
  }));

export default () =>{
    const classes = useStyles();
    return (
        <AppBar position="static">
            <Toolbar>
                <Typography variant="h6" className={classes.title}>
                InteliCam
                </Typography>
            </Toolbar>
        </AppBar>
    )
}
