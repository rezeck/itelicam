// import react modules
import React, {Component} from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { Grid, Card, CardContent,CardMedia, Typography} from '@material-ui/core'
import fetch from 'node-fetch'
import _ from 'lodash'


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


const userCard = (k)=>{
    return (<Card key={k} style={{margin:'12px'}}>
        <CardMedia
          component="img"
          alt=""
          height="140"
          image="https://www.materialui.co/materialIcons/action/face_black_144x144.png"
          title=""
        />
        <CardContent>
        <Typography gutterBottom variant="h5" component="h2">
            Pessoa
          </Typography>
          <Typography variant="body2" color="textSecondary" component="p">
           Não identificada
          </Typography>
        </CardContent>
    </Card>)
}

const NPERSONS_ENDPOINT = 'http://127.0.0.1:8000/npersons'
export default class extends Component{
    constructor(props){
        super(props)
        this.state = {
            n:0
        }
    }

    loadNPerson(){
        fetch(NPERSONS_ENDPOINT,{mode: 'cors'}).then( resp => resp.json()).then(resp =>{
            this.setState({
                n: resp.n_persons
            })
        } )
    }

    componentDidMount() {
        this.timerID = setInterval(
          () => this.loadNPerson(),
          1000
        );
      }
    
    componentWillUnmount() {
    clearInterval(this.timerID);
    }

    render(){
        return( <Grid container>
            {_.range(this.state.n).map((_,k)=> userCard(k))}
        </Grid>)
    } 

}